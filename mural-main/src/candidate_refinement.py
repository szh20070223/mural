"""A 初检不足时，合并来自原图和局部放大图的补查结果。"""

from collections import defaultdict
from itertools import combinations

from src.result_parser import MAX_CANDIDATES, parse_observation
from src.review_rules import iou
from src.schemas import Observation, ObservedRegion

MIN_CANDIDATES = 8
DUPLICATE_IOU = 0.85


def _duplicate(a: ObservedRegion, b: ObservedRegion) -> bool:
    return a.disease == b.disease and iou(a.bbox, b.bbox) >= DUPLICATE_IOU


def initial_candidates(observation: Observation) -> Observation:
    unique = []
    for finding in observation.findings:
        if finding.parent_region_id is not None:
            raise ValueError("初检条目不能引用父框；只有局部补查可拆分")
        if not any(_duplicate(finding, other) for other in unique):
            unique.append(finding)
    return observation.model_copy(update={"findings": unique})


def split_parents(initial: Observation) -> dict[str, ObservedRegion]:
    # 原图分辨率和画幅不同。优先放大较大区域，由 A 核对其内部是否有独立证据。
    typed = (f for f in initial.findings if f.disease != "待判读")
    return {f.region_id: f for f in sorted(typed, key=lambda f: -f.bbox.area_ratio())}


def _inside(inner, outer):
    return (outer.x1 <= inner.x1 < inner.x2 <= outer.x2
            and outer.y1 <= inner.y1 < inner.y2 <= outer.y2)


def apply_refinement(initial: Observation, text: str,
                     image_size: tuple[int, int] | None = None) -> Observation:
    """补查仅输出新增条目。未拆的初检项由程序保留，拆过的父框退出计数。"""
    delta, meta = parse_observation(text)
    if meta.truncated:
        raise ValueError("补查输出超过十条，请保留证据最明确的局部")
    parents = split_parents(initial)
    groups = defaultdict(list)
    additions = []
    for finding in delta.findings:
        parent_id = finding.parent_region_id
        if parent_id is None:
            if any(finding.disease == f.disease and _inside(finding.bbox, f.bbox)
                   for f in initial.findings):
                raise ValueError("原框内的同类证据必须填写 parent_region_id")
            additions.append(finding)
            continue
        parent = parents.get(parent_id)
        if not parent or parent.disease != finding.disease or not _inside(finding.bbox, parent.bbox):
            raise ValueError(f"{parent_id} 的子框必须在可拆父框内并与父框同类")
        if image_size:
            width, height = image_size
            box = finding.bbox
            if (box.x2 - box.x1) * width < 1 or (box.y2 - box.y1) * height < 1:
                raise ValueError(f"{parent_id} 子框小于一个原图像素")
        groups[parent_id].append(finding)

    combined = []
    for original in initial.findings:
        children = groups.get(original.region_id)
        if not children:
            combined.append(original)
            continue
        if len(children) < 2:
            raise ValueError(f"{original.region_id} 拆分至少需要两个局部证据框")
        for a, b in combinations(children, 2):
            if iou(a.bbox, b.bbox) > 0 or a.observation == b.observation:
                raise ValueError(f"{original.region_id} 的子框应各自定位、描述，不得重叠或复用观察文字")
        combined.extend(children)
    combined.extend(additions)
    if len(combined) > MAX_CANDIDATES:
        raise ValueError(f"合并后共 {len(combined)} 条，须将总数控制在十条内")
    if any(_duplicate(a, b) for a, b in combinations(combined, 2)):
        raise ValueError("补查重复框选了已有同类位置")
    numbered = [f.model_copy(update={"region_id": f"C{i}"}) for i, f in enumerate(combined, 1)]
    return Observation(image_id=initial.image_id, findings=numbered,
                       summary=delta.summary or initial.summary,
                       limitations=list(dict.fromkeys([*initial.limitations, *delta.limitations]))[:3])
