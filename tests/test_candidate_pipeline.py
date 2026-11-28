"""A/B 的离线回归用例。所有框、病名和模型回复均为测试输入。"""

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.candidate_refinement import apply_refinement, initial_candidates
from src.review_rules import apply_verdicts
from src.schemas import Observation, ReviewReport


def finding(i, disease="颜料层脱落", box=None):
    return dict(region_id=f"C{i}", disease=disease,
                bbox=box or [(i-1)/10, .1, (i-1)/10+.05, .18],
                observation=f"第{i}段衣纹见浅色缺口", impact="局部衣纹中断", confidence="中")


def batch(rows):
    return Observation(image_id="fixture", findings=rows)


def report(observation, grades, decisions=None):
    return ReviewReport(image_id="fixture", verdicts=[
        dict(region_id=f.region_id, decision=(decisions or {}).get(f.region_id, "keep"),
             reason="局部表面可见不规则边界", adjusted_confidence=grade)
        for f, grade in zip(observation.findings, grades)
    ])


def delta(children):
    return json.dumps(dict(image_id="fixture", findings=children), ensure_ascii=False)


class CandidateRules(unittest.TestCase):
    def setUp(self):
        self.initial = batch([finding(1, box=[0, .3, .9, .8])] + [finding(i) for i in range(2, 8)])
        self.children = [dict(finding(i, box=box), parent_region_id="C1") for i, box in
                         enumerate(([.01, .31, .15, .4], [.4, .5, .6, .7]), 1)]

    def test_split_replaces_parent_and_retains_other_six(self):
        refined = apply_refinement(self.initial, delta(self.children))
        self.assertEqual(len(refined.findings), 8)
        self.assertEqual([r.parent_region_id for r in refined.findings], ["C1", "C1"] + [None]*6)
        self.assertEqual(refined.findings[2].bbox, self.initial.findings[1].bbox)
        self.assertNotIn(self.initial.findings[0].bbox, [r.bbox for r in refined.findings])

    def test_invalid_splits(self):
        for change in [dict(bbox=[.4, .69, .7, .95]), dict(bbox=[.01, .31, .14, .39]),
                       dict(parent_region_id="C9"), dict(disease="水渍"),
                       dict(observation=self.children[0]["observation"])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                apply_refinement(self.initial, delta([self.children[0], dict(self.children[1], **change)]))
        with self.assertRaises(ValueError):
            apply_refinement(self.initial, delta(self.children[:1]))

    def test_duplicate_first_pass_not_counted(self):
        eight = batch([finding(i) for i in range(1, 8)] + [dict(finding(1), region_id="C8")])
        self.assertEqual(len(initial_candidates(eight).findings), 7)

    def test_subpixel_child_is_not_a_new_region(self):
        child = dict(self.children[0], bbox=[.01, .31, .0101, .4])
        with self.assertRaises(ValueError):
            apply_refinement(self.initial, delta([child, self.children[1]]), (320, 200))

    def test_independent_new_finding(self):
        updated = apply_refinement(self.initial, delta([finding(1, "水渍", [.1,.8,.2,.9])]))
        self.assertEqual(len(updated.findings), 8)
        self.assertEqual(updated.findings[-1].disease, "水渍")

    def test_unique_pair_all_same(self):
        obs = batch([finding(i, "颜色层脱落") for i in range(1, 9)])
        regions, stats = apply_verdicts(obs, report(obs, ["中"]*8))
        self.assertEqual([r.disease for r in regions], ["颜料层脱落"])
        self.assertEqual(stats.grades, {"高": 0, "中": 8, "低": 0})

    def test_low_with_typed_findings_and_high_crack(self):
        obs = batch([finding(i) for i in range(1, 8)] + [finding(8, "裂隙")])
        regions, stats = apply_verdicts(obs, report(obs, ["中"]*6 + ["低", "高"]))
        self.assertEqual(stats.selected_ids, ["C1", "C7", "C8"])
        self.assertEqual([(r.confidence, r.pending_review) for r in regions],
                         [("中", False), ("低", True), ("高", False)])

    def test_all_pending(self):
        obs = batch([finding(i, "待判读") for i in range(1, 9)])
        regions, _ = apply_verdicts(obs, report(obs, ["低"]*8))
        self.assertEqual(len(regions), 1)
        self.assertTrue(regions[0].pending_review)

    def test_global_diversity_and_priority(self):
        obs = batch([finding(1), finding(2), finding(3, "裂隙"), finding(4, "水渍")])
        regions, _ = apply_verdicts(obs, report(obs, ["中", "低", "中", "高"]))
        self.assertEqual([r.confidence for r in regions], ["低","中","高"])
        self.assertEqual(len({r.disease for r in regions}), 3)

    def test_missing_b_grade_or_candidate_fails(self):
        for grades in [[], [None]]:
            obs = batch([finding(1)])
            with self.assertRaises(ValueError):
                apply_verdicts(obs, report(obs, grades))

    def test_different_grades_not_lost_in_spatial_merge(self):
        obs = batch([finding(1), finding(2, box=[.005,.1,.05,.18])])
        regions, _ = apply_verdicts(obs, report(obs, ["中","低"]))
        self.assertEqual(len(regions), 2)

    def test_defer_does_not_inherit_an_unconfirmed_type(self):
        obs = batch([finding(1)])
        verdict = dict(region_id="C1", decision="defer", reason="留白和颜料缺失难以区分",
                       adjusted_observation="边缘有浅色斑块", adjusted_impact="衣纹末端难以判断",
                       adjusted_confidence="低")
        for disease, expected in [(None, "待判读"), ("颜料层脱落", "颜料层脱落")]:
            reviewed = ReviewReport(image_id="fixture", verdicts=[dict(verdict, adjusted_disease=disease)])
            regions, _ = apply_verdicts(obs, reviewed)
            self.assertEqual(regions[0].disease, expected)
            self.assertTrue(regions[0].pending_review)


class WorkflowTests(unittest.TestCase):
    def run_workflow(self, initial, additions=None, grades=None, *, drop=False, retry_grade=False):
        # Model calls are replaced at the transport boundary; real A/B prompts, parsing,
        # C orchestration, card builder and Pillow renderer run normally.
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline", "QWEN_MODEL": "offline"}):
            from src import workflow as w
        from src.qwen_client import ModelReply, QwenClient
        sequence = [delta(initial)]
        if additions is not None:
            sequence.append(delta(additions))
        if grades:
            observed = batch([finding(i+1) for i in range(len(grades))])
            decisions = {f.region_id: "drop" for f in observed.findings} if drop else {}
            reviewed = report(observed, grades, decisions)
            if retry_grade:
                invalid = reviewed.model_dump()
                invalid["verdicts"][0]["adjusted_confidence"] = None
                sequence.append(json.dumps(invalid))
            sequence.append(reviewed.model_dump_json())
            if not drop:
                sequence.append(json.dumps(dict(summary="局部可见衣纹中断。", limitations=[],
                    recommendations=[dict(action="记录局部并放大比对", basis="R1 的衣纹中断可对照",
                                          risk="补光可能影响颜色")])))
        replies = [ModelReply(t, "offline", 0, 0, 0, "stop") for t in sequence]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            photo = root / "fixture.png"
            Image.new("RGB", (320, 200), "white").save(photo)
            for attr in ("RAW_DIR","OBSERVATION_DIR","REVIEW_DIR","CARD_DIR"):
                stack.enter_context(patch.object(w, attr, root / attr))
            stack.enter_context(patch("src.trace.TRACE_DIR", root / "logs"))
            transport = stack.enter_context(patch.object(QwenClient, "chat", side_effect=replies))
            data = w.run_vision_workflow(str(photo), image_source="离线测试图", image_license="测试")
            self.assertEqual(transport.call_count, len(replies))
            if data["card"]:
                self.assertGreater(Path(data["card"]["pdf_path"]).stat().st_size, 0)
                serialized = json.loads(Path(data["card"]["json_path"]).read_text(encoding="utf-8"))
                self.assertEqual(serialized["regions"], data["review"]["regions"])
            if data["review"]:
                applied = json.loads(Path(data["review"]["applied_path"]).read_text(encoding="utf-8"))
                self.assertEqual(applied["decisions"], data["review"]["decisions"])
            return data

    def test_seven_to_eight_to_three(self):
        initial = [finding(1, box=[0,.3,.9,.8])] + [finding(i) for i in range(2,7)] + [finding(7,"裂隙")]
        part = [dict(finding(i, box=b), parent_region_id="C1") for i,b in
                enumerate(([.01,.31,.15,.4],[.4,.5,.6,.7]),1)]
        data = self.run_workflow(initial, part, ["中"]*6 + ["低","高"], retry_grade=True)
        self.assertEqual(data["candidate_status"]["count"],8)
        self.assertEqual(data["review"]["attempts"],2)
        self.assertEqual(len(data["review"]["decisions"]),8)
        self.assertEqual(data["review"]["stats"]["final"],3)

    def test_eight_to_one_no_refinement(self):
        data = self.run_workflow([finding(i) for i in range(1,9)], grades=["中"]*8)
        self.assertEqual(data["candidate_status"]["refinement_attempts"],0)
        self.assertEqual(len(data["review"]["regions"]),1)

    def test_no_evidence_and_partial_batch(self):
        for initial in [[], [finding(1)]]:
            data = self.run_workflow(initial, [])
            self.assertFalse(data["candidate_status"]["ready"])
            self.assertIsNone(data["review"])
            self.assertIsNone(data["card"])

    def test_all_drop_keeps_decisions_and_no_card(self):
        data = self.run_workflow([finding(i) for i in range(1,9)], grades=["低"]*8, drop=True)
        self.assertIsNone(data["card"])
        self.assertEqual(len(data["review"]["decisions"]), 8)
        self.assertTrue(all(r["pending_review"] for r in data["review"]["decisions"]))


if __name__ == "__main__":
    unittest.main()
