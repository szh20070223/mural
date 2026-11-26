"""离线回归：只验证工程行为，模拟回复不代表模型识别效果。"""
import base64
import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from src.images import load_image
from src.qwen_client import ModelReply, QwenClient, _looks_like_unsupported_param
from src.result_parser import extract_json, parse_observation, parse_report, parse_review
from src.review_rules import apply_verdicts, validate_review
from src.schemas import AnnotationRecord, Card, Observation, ReviewReport, ReviewVerdict
from src.workflow import run_vision_workflow
from src.render import render_card

ROOT = Path(__file__).resolve().parents[1]


def finding(number=1, **changes):
    result = dict(region_id=f"C{number}", disease="颜料层脱落", bbox=[.1, .1, .4, .4],
                  observation="浅色缺块露出下层", impact="人物轮廓局部中断", confidence="高", limitations=[])
    result.update(changes)
    return result


def observation(*items):
    return Observation(image_id="test", findings=list(items) or [finding()], limitations=["缺少侧光图"])


def verdict(number=1, **changes):
    result = dict(region_id=f"C{number}", decision="keep", reason="可见局部边界需核实", review_notes=["补拍局部与侧光照片"])
    result.update(changes)
    return result


def review(*items):
    return ReviewReport(image_id="test", verdicts=list(items), reviewer_notes=["原绘白粉仍需排除"])


def draft():
    return dict(summary="局部浅色区域值得进一步核查", recommendations=[dict(
        action="补拍侧光和局部照片", basis="R1 的浅色边界需进一步核实", risk="补光可能改变视觉色差",
        needs_human_review=True)], limitations=["照片不足以鉴定病害"])


def reply(data, model="actual-model"):
    return ModelReply(json.dumps(data, ensure_ascii=False), model, 1, 10, 20, "stop")


class ParserTests(unittest.TestCase):
    def test_missing_findings_are_not_negative(self):
        for bad in ({"image_id": "x"}, {"image_id": "x", "findings": None}, {"findings": "none"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_observation(json.dumps(bad))
        result, _ = parse_observation('{"image_id":"x","findings":[]}')
        self.assertEqual(result.findings, [])

    def test_json_repair_preserves_evidence(self):
        self.assertEqual(extract_json('{"text":"引号“原绘”,}和,]", "items":[1,],}')['text'], '引号“原绘”,}和,]')

    def test_bbox_not_guessed(self):
        with self.assertRaises(ValueError):
            parse_observation(json.dumps(dict(image_id="x", findings=[finding(bbox=[.1, .1, 1.2, .8])])))
        result, meta = parse_observation(json.dumps(dict(image_id="x", coordinate_scale="thousand",
            findings=[finding(bbox=[100, 100, 400, 400])])))
        self.assertEqual(result.findings[0].bbox.x2, .4)
        self.assertEqual(meta.bbox_rescaled, 1)

    def test_duplicate_candidate_rejected(self):
        with self.assertRaises(ValueError):
            observation(finding(), finding())

    def test_missing_review_array_rejected(self):
        with self.assertRaises(ValueError):
            parse_review('{"image_id":"x","verdicts":null}')

    def test_reference_must_exist(self):
        for reference in ("R10", "R2"):
            data = draft(); data['recommendations'][0]['basis'] = f"{reference} 区域需要核实"
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                parse_report(json.dumps(data), {"R1"})

    def test_empty_risk_after_prefix_removal_rejected(self):
        data = draft(); data['recommendations'][0]['risk'] = "风险："
        with self.assertRaises(ValueError):
            parse_report(json.dumps(data), {"R1"})

    def test_model_cannot_waive_human_review(self):
        data = draft(); data['recommendations'][0]['needs_human_review'] = False
        with self.assertRaises(ValueError):
            parse_report(json.dumps(data), {"R1"})

    def test_whitespace_observation_rejected(self):
        with self.assertRaises(ValueError):
            observation(finding(observation='     '))


class ReviewTests(unittest.TestCase):
    def test_revised_category_requires_consistent_text(self):
        with self.assertRaises(ValueError):
            ReviewVerdict(**verdict(adjusted_disease="水渍"))

    def test_defer_replaces_assertion(self):
        regions, _ = apply_verdicts(observation(), review(verdict(decision="defer",
            adjusted_observation="局部可见浅色斑块", adjusted_impact="局部轮廓与色差待辨")))
        self.assertEqual(regions[0].disease, "待判读")
        self.assertEqual(regions[0].confidence, "低")
        self.assertEqual(regions[0].observation, "局部可见浅色斑块")
        self.assertEqual(regions[0].source_candidate_ids, ["C1"])

    def test_adjust_can_change_confidence_without_bbox(self):
        regions, _ = apply_verdicts(observation(), review(verdict(decision="adjust", adjusted_confidence="低")))
        self.assertEqual(regions[0].confidence, "低")

    def test_missing_and_unknown_verdict_rejected(self):
        for report in (review(), review(verdict(2))):
            with self.assertRaises(ValueError):
                apply_verdicts(observation(), report)

    def test_duplicate_verdict_rejected(self):
        with self.assertRaises(ValueError):
            review(verdict(), verdict())

    def test_conflicting_diseases_remain_separate(self):
        regions, stats = apply_verdicts(observation(finding(), finding(2, disease="水渍")),
                                        review(verdict(), verdict(2)))
        self.assertEqual(len(regions), 2)
        self.assertEqual(stats.merged_by_iou, 0)

    def test_iou_merge_does_not_boost_confidence(self):
        regions, _ = apply_verdicts(observation(finding(confidence="低"), finding(2)),
                                    review(verdict(), verdict(2)))
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].confidence, "低")
        self.assertEqual(regions[0].source_candidate_ids, ["C1", "C2"])

    def test_invalid_merge_targets_rejected(self):
        obs = observation(finding(), finding(2))
        for target in (verdict(2, decision="drop"), verdict(2, decision="merge", merge_into="C1")):
            with self.assertRaises(ValueError):
                validate_review(obs, review(verdict(decision="merge", merge_into="C2"), target))

    def test_nonoverlapping_merge_rejected(self):
        obs = observation(finding(), finding(2, bbox=[.7, .7, .9, .9]))
        with self.assertRaises(ValueError):
            validate_review(obs, review(verdict(decision="merge", merge_into="C2"), verdict(2)))

    def test_review_priority_controls_card_order(self):
        regions, _ = apply_verdicts(observation(finding(), finding(2, bbox=[.7, .7, .9, .9])),
                                    review(verdict(2), verdict()))
        self.assertEqual(regions[0].source_candidate_ids, ["C2"])


class InputTests(unittest.TestCase):
    def test_uncertain_image_not_negative(self):
        data = dict(image_id="x", source="s", license="l", width=10, height=10,
                    assessment="无法判断", negative=False, regions=[])
        self.assertEqual(AnnotationRecord(**data).assessment, "无法判断")
        data['negative'] = True
        with self.assertRaises(ValueError):
            AnnotationRecord(**data)

    def test_orientation_and_transparency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'oriented.jpg'
            photo = Image.new('RGB', (30, 10), 'red')
            exif = Image.Exif(); exif[274] = 6
            photo.save(path, exif=exif)
            self.assertEqual(load_image(path).size, (10, 30))
            path = path.with_suffix('.png')
            Image.new('RGBA', (10, 10), (0, 0, 0, 0)).save(path)
            self.assertEqual(load_image(path).getpixel((0, 0)), (255, 255, 255))

    def test_bmp_becomes_png_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.bmp'
            Image.new('RGB', (30, 10), 'red').save(path)
            client = QwenClient.__new__(QwenClient); client._image_cache = None
            encoded = client.encode_image(str(path))
            self.assertTrue(encoded.startswith('data:image/png;base64,'))
            image = Image.open(io.BytesIO(base64.b64decode(encoded.split(',', 1)[1])))
            self.assertEqual(image.size, (30, 10))

    def test_parameter_fallback_is_specific(self):
        error = RuntimeError('invalid image parameter'); error.status_code = 400
        self.assertFalse(_looks_like_unsupported_param(error))
        error = RuntimeError('unsupported enable_thinking'); error.status_code = 400
        self.assertTrue(_looks_like_unsupported_param(error))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.image = self.directory / 'sample.bmp'
        Image.new('RGB', (320, 180), 'beige').save(self.image)
        for name in ('RAW_DIR', 'OBSERVATION_DIR', 'REVIEW_DIR', 'CARD_DIR', 'INPUT_DIR'):
            self.stack.enter_context(patch(f'src.workflow.{name}', self.directory / name))
        self.stack.enter_context(patch('src.trace.TRACE_DIR', self.directory / 'trace'))
        self.client = Mock(model='configured-alias', thinking_enabled=False)
        self.stack.enter_context(patch('src.workflow.QwenClient', return_value=self.client))
        self.reset_replies()

    def reset_replies(self, reviews=None):
        self.client.analyze_image.side_effect = [reply(observation().model_dump(), 'actual-a')] + (
            reviews if reviews is not None else [reply(review(verdict()).model_dump(), 'actual-b')])
        self.client.chat.return_value = reply(draft(), 'actual-c')

    def test_no_render_and_provenance(self):
        with patch('src.workflow.render_card') as renderer:
            result = run_vision_workflow(str(self.image), render=False, scene_note="拍摄背景")
            renderer.assert_not_called()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['card']['render_status'], 'skipped')
        card = Card.model_validate_json(Path(result['card']['json_path']).read_text(encoding='utf-8'))
        self.assertEqual(card.provenance.models, ['actual-a', 'actual-b', 'actual-c'])
        self.assertEqual(len(card.provenance.input_sha256), 64)
        self.assertIn('原绘白粉仍需排除', '；'.join(card.limitations))
        self.assertIn('缺少侧光图', '；'.join(card.limitations))
        self.assertIn('拍摄背景', self.client.chat.call_args.args[0][0]['content'])
        self.assertEqual(card.attribution.human_reviewed, [])

    def test_repeat_runs_preserve_previous_outputs(self):
        first = run_vision_workflow(str(self.image), render=False)
        self.reset_replies()
        second = run_vision_workflow(str(self.image), render=False)
        self.assertNotEqual(first['output_path'], second['output_path'])
        self.assertTrue(Path(first['output_path']).is_file())
        self.assertTrue(Path(second['input_path']).is_file())

    def test_parallel_runs_preserve_artifacts(self):
        def factory():
            client = Mock(model='configured-alias', thinking_enabled=False)
            client.analyze_image.side_effect = [reply(observation().model_dump()), reply(review(verdict()).model_dump())]
            client.chat.return_value = reply(draft())
            return client
        with patch('src.workflow.QwenClient', side_effect=factory), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: run_vision_workflow(str(self.image), render=False), range(2)))
        self.assertNotEqual(results[0]['run_id'], results[1]['run_id'])
        for result in results:
            self.assertTrue(Path(result['card']['json_path']).is_file())

    def test_disabled_review_and_composition(self):
        result = run_vision_workflow(str(self.image), review=False)
        self.assertEqual(result['status'], 'completed')
        self.assertIsNone(result['card'])
        self.assertEqual(self.client.analyze_image.call_count, 1)
        self.client.chat.assert_not_called()

    def test_no_candidates_no_card(self):
        self.client.analyze_image.side_effect = [reply(dict(image_id='x', findings=[], limitations=[]))]
        result = run_vision_workflow(str(self.image))
        self.assertEqual(result['status'], 'no_card')
        self.assertEqual(self.client.analyze_image.call_count, 1)

    def test_length_limited_reply_not_accepted(self):
        truncated = reply(observation().model_dump()); truncated.finish_reason = 'length'
        self.client.analyze_image.side_effect = [truncated, reply(observation().model_dump()), reply(review(verdict()).model_dump())]
        result = run_vision_workflow(str(self.image), render=False)
        self.assertEqual(result['attempts'], 2)

    def test_incomplete_review_retries(self):
        self.reset_replies([reply(review().model_dump()), reply(review(verdict()).model_dump())])
        result = run_vision_workflow(str(self.image), render=False)
        self.assertEqual(result['review']['attempts'], 2)
        self.assertEqual(self.client.analyze_image.call_count, 3)

    def test_all_dropped_does_not_claim_complete_card(self):
        self.reset_replies([reply(review(verdict(decision='drop')).model_dump())])
        result = run_vision_workflow(str(self.image), render=False)
        self.assertEqual(result['status'], 'no_card')
        self.assertIsNone(result['card'])
        self.client.chat.assert_not_called()
        log = [json.loads(line) for line in Path(result['trace_path']).read_text(encoding='utf-8').splitlines()]
        self.assertEqual(log[-1]['regions'], 0)
        self.assertFalse(log[-1]['ok'])

    def test_failure_has_end_event(self):
        self.client.analyze_image.side_effect = RuntimeError('offline test failure')
        with self.assertRaises(RuntimeError):
            run_vision_workflow(str(self.image))
        logpath = next((self.directory / 'trace').glob('*.jsonl'))
        last = json.loads(logpath.read_text(encoding='utf-8').splitlines()[-1])
        self.assertEqual(last['event'], 'run_end')
        self.assertFalse(last['ok'])

    def test_render_failure_keeps_json_and_status(self):
        with patch('src.workflow.render_card', side_effect=ValueError('overflow')):
            result = run_vision_workflow(str(self.image))
        self.assertEqual(result['status'], 'render_failed')
        self.assertTrue(Path(result['card']['json_path']).is_file())

    def test_complete_pipeline_renders(self):
        result = run_vision_workflow(str(self.image))
        self.assertEqual(result['status'], 'completed', result['reason'])
        self.assertTrue(Path(result['card']['pdf_path']).is_file())


class RenderTests(unittest.TestCase):
    def test_oversize_text_is_not_silently_cut(self):
        card = Card.model_validate_json((ROOT / 'docs/card.example.json').read_text(encoding='utf-8'))
        card.provenance.image_source = '来源说明' * 2000
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'image.png'; pdf = Path(directory) / 'card.pdf'
            Image.new('RGB', (200, 100), 'beige').save(image)
            with self.assertRaises(ValueError):
                render_card(card, image, pdf)
            self.assertFalse(pdf.exists())


if __name__ == '__main__':
    unittest.main()
