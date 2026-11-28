# src/schemas.py
# =============================================================================
# 全队共用的数据契约（唯一真相源）
# =============================================================================
# 这个文件回答一个问题：我们的流水线最终要交出什么形状的数据。
#
# 为什么单独抽一个文件、而且写得这么啰嗦：
#   1. 有三拨人要同时消费它的输出——卡片渲染、PDF 报告、演示视频，
#      大家如果各自猜字段名，改一次 prompt 就要返工三次。
#   2. 比赛评委要抽查"哪些是 AI 生成、哪些人工复核过"。这两个信息必须
#      是结构化的字段，不能是散文里的一句话，否则没法核验。
#   3. 一页纸放不下太多字。这里每个字段都有字数上限，上限不是拍脑袋，
#      是按 A4 一页的排版预算倒推的（见 docs/card-contract.md）。
#
# 改动规则（很重要，别绕过）：
#   - 任何字段的增删改名，必须同时改三处：
#     ① 本文件 ② docs/card-contract.md ③ docs/card.example.json
#   - 改完跑一次：python tools/check_contract.py
#     示例跑不过 = 契约写错了，不是示例写错了。
#   - 下游（渲染器 / 报告 / 视频）只认 Card，不认 Observation 和旧版。
#
# 分层（从中间产物到最终交付物）：
#   A → Observation（初检/补查候选），B → ReviewReport（逐条分级和裁决）。
#   入选的 1–3 个区域交给 C 成稿，最后组装 Card。
#   旧版无框 AnalysisResult 仅用于回看和提示词对照。
# =============================================================================

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# =============================================================================
# 0. 常量
# =============================================================================

# 病害类型白名单。
# 为什么要定死这 8 个而不是让模型自由发挥：
#   - 模型如果一会儿写"褪色"、一会儿写"变色或褪色"，标注和评测就没法对齐；
#   - 卡片上的标签也只有定死才能做统一的配色和图标。
# 现场赛题面提到的病害是"裂隙、脱落、褪色、污染"等，这 8 类是它的细化。
# 如果之后要加类型，加在这里 + 同步 docs/annotation.md，不要临时拼字符串。
DiseaseType = Literal[
    "裂隙",
    "起甲",
    "颜料层脱落",
    "酥碱",
    "粉化",
    "水渍",
    "变色或褪色",
    "地仗层脱落",
]

DISEASE_TYPES: tuple[str, ...] = (
    "裂隙",
    "起甲",
    "颜料层脱落",
    "酥碱",
    "粉化",
    "水渍",
    "变色或褪色",
    "地仗层脱落",
)

# 别名表：把模型常用的同义词收敛到白名单里的规范名。
# 只做"整串完全相等"的匹配，不做模糊替换——
# 模糊替换有风险，比如"地仗层脱落"里含"脱落"，一旦乱替换就会串味。
# 左右两边都手写，看到新的说法就往里加一行，别在代码里另开分支。
DISEASE_ALIASES: dict[str, str] = {
    "龟裂": "裂隙",
    "裂缝": "裂隙",
    "裂纹": "裂隙",
    "网状裂隙": "裂隙",
    "起翘": "起甲",
    "颜料层起甲": "起甲",
    "颜色层脱落": "颜料层脱落",
    "待判断": "待判读",
    "颜料脱落": "颜料层脱落",
    "颜色脱落": "颜料层脱落",
    "剥落": "颜料层脱落",
    "泛碱": "酥碱",
    "盐析": "酥碱",
    "起霜": "酥碱",
    "粉状化": "粉化",
    "水痕": "水渍",
    "水渍痕": "水渍",
    "褪色": "变色或褪色",
    "变色": "变色或褪色",
    "色变": "变色或褪色",
    "地仗脱落": "地仗层脱落",
    "地层脱落": "地仗层脱落",
}


def _normalize_disease(value: object) -> object:
    """把模型/标注里的病害名收敛成白名单里的规范名。

    mode="before" 的校验器，所以这里拿到的还是"原始输入"，可能是 None、
    可能是带空格的字符串。非字符串直接放行，交给后面的类型检查去报错，
    这里不做兜底，免得把真正的错误吞掉。
    """
    if isinstance(value, str):
        cleaned = value.strip()
        return DISEASE_ALIASES.get(cleaned, cleaned)
    return value


# 8 类真实病害：标注、评测用这个。
DiseaseName = Annotated[DiseaseType, BeforeValidator(_normalize_disease)]

# 9 类 = 8 个真实病害 + "待判读"。
#
# "待判读"不是第九种病害，是"这处确实有异常、但类型判断不了"这个**结论**。
# 只给模型输出用（A 的候选、B 的改判、卡片上的区域）；
# 标注集不能用它——标注是标准答案，"看不清"应该走 assessment 字段。
#
# 为什么要分成两个类型而不是加进 DiseaseType 再排除：
#   加进去之后，标注校验就得额外维护一份"排除表"，那种补丁迟早会漏。
#   分开定义，标注那边天然只有 8 个，不需要任何排除逻辑。
ObservationDisease = Annotated[
    DiseaseType | Literal["待判读"],
    BeforeValidator(_normalize_disease),
]


def _normalize_region_id(value: object, prefix: str) -> object:
    """把各种写法的编号收敛成 C1 / R1。

    模型对编号的写法相当随意：可能写 "1"、"c1"、"C-1"、"R1区"。
    编号只用于排序和跟图上标签对应，不影响任何判断，
    所以这里尽力修正而不是直接报错——为这点小事重试一次是浪费钱。
    修不了的（比如 "左上角那块"）照原样返回，交给 pattern 校验报错。
    """
    if not isinstance(value, str):
        return value

    cleaned = value.strip().upper()

    # "1" -> "C1"（只有纯数字时才补前缀）
    if cleaned.isdigit():
        return f"{prefix}{cleaned}"

    # "C-1" / "C 1" -> "C1"
    if cleaned.startswith(prefix):
        rest = cleaned[len(prefix):].lstrip("-— _")
        if rest.isdigit():
            return f"{prefix}{rest}"

    return cleaned

# 置信度：卡片上只留三档。
# 注意：没有"无法评估"——因为能进卡片的区域必须是"值得关注的区域"，
# 如果一个区域连判断都做不了，它就不该占用 1-3 个名额里的位置，
# 应该写进 limitations 里说明"这张图有哪些病害看不出来"。
RegionConfidence = Literal["高", "中", "低"]

# 全卡统一的免责声明。
# 这句不是客套，是赛题《安全及风险提示》第 4 条的要求：
# "生成式 AI 输出须经人工复核，参赛队伍并标注其中存在的不确定性"。
# 写在代码里而不是让模型生成，是为了保证每张卡一字不差——
# 模型自由发挥的免责声明，措辞每次都不一样，反而显得不专业。
DEFAULT_DISCLAIMER = (
    "本卡为 AI 辅助观察结果，已完成/待完成人工复核，仅用于数字展示与"
    "辅助说明，不构成文物病害的专业鉴定或修复施工依据；"
    "涉及修复方案的内容须由专业人员现场判断。"
)


# =============================================================================
# 1. 基础类型：归一化坐标框
# =============================================================================


class BoundingBox(BaseModel):
    """一个矩形框，坐标全部是 0-1 的归一化值。

    为什么用归一化坐标、不用像素坐标：
      - 视觉模型看不到图片的原始分辨率，问它像素它只能猜；
        问它"占画面从左到右百分之多少"，它反而答得准。
      - 归一化之后换分辨率不用改提示词、不用重标数据，
        渲染时乘一下宽高就回到像素（见 to_pixels）。
    坐标约定：原点在左上角，x 向右增大，y 向下增大。
    这是 PIL / OpenCV / 所有视觉模型的共同约定，别反了。
    """

    x1: float = Field(ge=0.0, le=1.0, description="左边界，0 表示画面最左")
    y1: float = Field(ge=0.0, le=1.0, description="上边界，0 表示画面最上")
    x2: float = Field(ge=0.0, le=1.0, description="右边界，1 表示画面最右")
    y2: float = Field(ge=0.0, le=1.0, description="下边界，1 表示画面最下")

    @model_validator(mode="before")
    @classmethod
    def _from_array(cls, value: object) -> object:
        """允许模型直接吐数组 [x1, y1, x2, y2]。

        模型输出对象 {"x1": ...} 时嵌套层数多，出错概率比数组高，
        而且数组少打几个花括号，token 也更省。所以提示词里让模型写数组，
        这里统一转成字段。两种写法都能通过校验，互不干扰。
        """
        if isinstance(value, (list, tuple)):
            if len(value) != 4:
                raise ValueError(
                    f"bbox 必须是 4 个数 [x1, y1, x2, y2]，收到 {len(value)} 个"
                )
            return dict(zip(("x1", "y1", "x2", "y2"), value))
        return value

    @model_validator(mode="after")
    def _check_order(self) -> "BoundingBox":
        """左上角必须严格小于右下角。

        模型偶尔会把两对坐标写反（先给右下再给左上），
        这种错误不会抛异常、只会画出一个空框，肉眼还不容易发现，
        所以在校验层直接拦掉，让它变成一次可重试的失败。
        """
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError(
                "bbox 左上角必须小于右下角，"
                f"收到 [{self.x1}, {self.y1}, {self.x2}, {self.y2}]"
            )
        return self

    def as_list(self) -> list[float]:
        """还原成 [x1, y1, x2, y2]，方便直接写回 JSON。"""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """换算成 PIL 能直接画的像素坐标 (left, top, right, bottom)。

        顺手处理两个边界情况：
          - 四舍五入后可能塌成一条线，所以保证宽高至少 1 像素；
          - 模型可能给出 0.999 这种值，乘完再取整会越界，所以夹到图内。
        """
        x1 = min(max(int(round(self.x1 * width)), 0), width - 1)
        y1 = min(max(int(round(self.y1 * height)), 0), height - 1)
        x2 = min(max(int(round(self.x2 * width)), x1 + 1), width)
        y2 = min(max(int(round(self.y2 * height)), y1 + 1), height)
        return x1, y1, x2, y2

    def area_ratio(self) -> float:
        """框占全图的比例，用来做"框是不是画得太粗"的软检查。

        比如一个框占了 80% 的画面，它其实等于没定位，
        渲染出来就是整张图被圈住，评委一眼就看出没用。
        这个值不参与校验（不抛错），只给 tools 里的检查脚本提示用。
        """
        return (self.x2 - self.x1) * (self.y2 - self.y1)


# =============================================================================
# 2. 中间产物：A 观察官 / B 审查官 之间的流转数据
# =============================================================================


class ObservedRegion(BaseModel):
    """A 观察官吐出的一个候选区域。

    和最终的 Region 的区别：一批最多 10 个，允许低置信度，
    因为 A 的职责是"尽量别漏"，B 的职责才是"砍到只剩站得住的"。
    如果把 A 也限制成 3 个，等于让模型在只看一次的情况下就做取舍，
    那是审查官该干的事，不是观察官该干的事。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    region_id: str = Field(pattern=r"^C([1-9]|1[0-2])$", description="候选编号 C1-C10")
    parent_region_id: str | None = Field(default=None, pattern=r"^C([1-9]|1[0-2])$",
                                         description="拆分来源，引用初检阶段的编号")
    disease: ObservationDisease
    bbox: BoundingBox
    observation: str = Field(
        min_length=4,
        max_length=60,
        description="看到了什么，必须是可指向的视觉事实。"
        "上限和 Region 一致：A 直接写成卡片能用的长度，B 就不用再做压缩，"
        "少一道压缩就少一次信息走样",
    )
    impact: str = Field(
        min_length=4,
        max_length=40,
        description="对画面内容的影响：挡住了什么、让哪部分无法辨认。"
        "这一句必须由 A 写——整条链路只有 A 看得到图片，"
        "留给只看文本的 C 去补，等于让没见过图的人描述图",
    )
    confidence: RegionConfidence
    # 候选项允许带自己的局限说明，B 会用它来决定要不要保留
    limitations: list[str] = Field(default_factory=list, max_length=2)

    @field_validator("region_id", mode="before")
    @classmethod
    def _fix_region_id(cls, value: object) -> object:
        """候选编号写歪了自动纠正，规则见 _normalize_region_id。"""
        return _normalize_region_id(value, "C")


class Observation(BaseModel):
    """A 观察官的完整输出（中间产物，不是交付物）。

    注意两点：
      - findings 允许为空：图里真的什么都没有时，A 应该能说"没看出问题"，
        而不是被迫编一个出来。这一点直接关系到误报率，别改成 min_length=1。
      - 这里没有 summary 以外的成稿字段，建议、来源、归属都归 C 管，
        各司其职，不要让 A 越界去写保护建议——它没有那个信息基础。
    """

    image_id: str
    findings: list[ObservedRegion] = Field(default_factory=list, max_length=10)
    limitations: list[str] = Field(default_factory=list, max_length=3)
    summary: str = Field(default="", max_length=120)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Observation":
        """候选编号不能重复。

        重复的后果很隐蔽：B 的裁决是按编号映射的，两个 C1 只有一个能命中，
        另一个会被当成"漏判"保守保留——日志里只多一个数字，看不出来。
        """
        ids = [item.region_id for item in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError(f"候选 region_id 不能重复：{ids}")
        return self


class ReviewVerdict(BaseModel):
    """B 审查官对一个候选区域的裁决（也是一份可展示的证据）。

    B 的价值不在"再看一遍"，而在于它必须**看图**并给出可核对的裁决理由。
    答辩时被问"你怎么保证 AI 没有瞎编"，拿出来的就是这一串记录：
    哪个候选被删了、为什么删、哪个框被改小了、改的依据是什么。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    region_id: str = Field(
        pattern=r"^C([1-9]|1[0-2])$",
        description="被裁决的候选编号，必须和 A 给出的编号一致",
    )
    # keep=保留  adjust=改判后保留  defer=保留位置但类型待判
    # drop=剔除  merge=与另一个候选合并
    decision: Literal["keep", "adjust", "drop", "merge", "defer"]
    reason: str = Field(
        min_length=4,
        max_length=80,
        description="裁决理由。必须说清'图上哪里不对'或'为什么成立'，"
        "只写'证据不足'不算理由——那是结论，不是依据",
    )
    # 改判文字：任何裁决都能用，不只是 adjust。
    # 这是"B 的结论能传导到卡片"的唯一通道——没有它，
    # B 就算判定"性质待定"，卡片上写的仍然是 A 那句"露出下层底色"。
    adjusted_observation: str | None = Field(default=None, min_length=4, max_length=60)
    adjusted_impact: str | None = Field(default=None, min_length=4, max_length=40)
    # adjust 时给新框；其余情况留空
    adjusted_bbox: BoundingBox | None = None
    # 改判病害类型时才填。拿不准就留空，在 reason 里说明
    adjusted_disease: ObservationDisease | None = None
    # 新运行要求 B 对全部候选显式分级。可空用于读取历史审查文件。
    adjusted_confidence: RegionConfidence | None = None
    # merge 时必填：合并到哪一个候选
    merge_into: str | None = Field(
        default=None,
        pattern=r"^C([1-9]|1[0-2])$",
        description="merge 时必填，指向被合并到的那个候选编号",
    )
    # 这一条进卡时，需要人复核的具体点。建议每条都写，空着说明审得不细
    review_notes: list[str] = Field(default_factory=list, max_length=2)

    @field_validator("region_id", "merge_into", mode="before")
    @classmethod
    def _fix_region_id(cls, value: object) -> object:
        """候选编号写歪了自动纠正，规则见 _normalize_region_id。"""
        return _normalize_region_id(value, "C")

    @model_validator(mode="after")
    def _check_decision_payload(self) -> "ReviewVerdict":
        """裁决和附件字段必须自洽。

        这几条校验都是为了拦住"看起来审了、其实没审"：
          - adjust 一个 adjusted_* 都不给 -> 等于没调整
          - defer 不说清看得见什么 -> 等于把"不知道"直接印在卡片上
          - merge 不指定目标 / 合并到自己 -> 不知道并到哪
        这些错误在渲染阶段很难发现，必须在这里拦住。
        """
        # adjust 至少要真的调整了点什么，否则等于没调整
        if self.decision == "adjust" and not any(
            value is not None
            for value in (
                self.adjusted_bbox,
                self.adjusted_disease,
                self.adjusted_confidence,
                self.adjusted_observation,
                self.adjusted_impact,
            )
        ):
            raise ValueError(
                f"{self.region_id}: decision=adjust 至少要给出一个 adjusted_* 字段"
            )

        # 改类型（或 defer）必须同时改写文字。
        # 只改类型不改描述，卡片上就会留下"水渍：……露出灰褐色下层"
        # 这种前后打架的话——那正是 ulti 那张卡的另一半病根。
        if self.adjusted_disease is not None or self.decision == "defer":
            if not self.adjusted_observation or not self.adjusted_impact:
                raise ValueError(
                    f"{self.region_id}: 改判类型或 defer 时，必须同时给出"
                    "中性的 adjusted_observation 和 adjusted_impact"
                )

        if self.decision == "merge":
           
            if self.merge_into is None:
                raise ValueError(f"{self.region_id}: decision=merge 必须给出 merge_into")
            if self.merge_into == self.region_id:
                raise ValueError(f"{self.region_id}: 不能合并到自己")
        elif self.merge_into is not None:
            # 非 merge 决策带 merge_into，数据里就留下一条没人执行的指令，
            # apply_verdicts 也会忽略它——不如在门口拦掉
            raise ValueError(
                f"{self.region_id}: 只有 decision=merge 才能指定 merge_into"
            )
        return self


class ReviewReport(BaseModel):
    """B 审查官的完整输出。

    保留被剔除的候选（dropped）而不是直接丢掉，是为了让整条链路可追溯。
    这条数据同时喂给三个地方：卡片的"人工复核提示"、演示视频的中间画面、
    以及 eval 脚本统计"模型提出了多少、审查砍掉了多少"。
    """

    image_id: str
    verdicts: list[ReviewVerdict] = Field(default_factory=list)
    reviewer_notes: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ReviewReport":
        """裁决编号不能重复。

        重复裁决会让同一条候选被处理两次（计数翻倍、区域被合并掉一次），
        而且 B 自己不会觉得有问题。在数据层拦掉，触发一次重试。
        """
        ids = [item.region_id for item in self.verdicts]
        if len(ids) != len(set(ids)):
            raise ValueError(f"裁决 region_id 不能重复：{ids}")
        return self


# =============================================================================
# 3. 最终交付物：一页观察卡
# =============================================================================


class Region(BaseModel):
    """卡片上标出的一个"值得关注的区域"。

    字数上限的由来（详见 docs/card-contract.md 的排版预算）：
    一页 A4 要放 1 张缩略图 + 3 个区域说明 + 2 条建议 + 局限性 + 来源，
    留给三个区域的正文一共只有 400 字左右，平均每个区域 130 字，
    所以 observation 60 + impact 40 + 两条复核提示各 30，刚好卡住。
    超过这个数，渲染器就必须缩字号或者溢出——两种都很难看。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    region_id: str = Field(pattern=r"^R[1-9]$", description="卡片上的编号 R1-R3")

    disease: ObservationDisease

    bbox: BoundingBox

    observation: str = Field(
        min_length=4,
        max_length=60,
        description="看到了什么。只写视觉事实：颜色、边界、纹理、走向，"
        "不要写'疑似病害严重'这种结论式套话",
    )

    impact: str = Field(
        min_length=4,
        max_length=40,
        description="对画面内容的影响。现场赛明确要求"
        "'说明其对画面内容的影响'，所以要写清楚挡住了什么、"
        "让哪部分形象无法辨认，而不是重复一遍病害名",
    )

    confidence: RegionConfidence

    pending_review: bool = Field(default=False, description="低置信度区域需要人工复核")

    @model_validator(mode="after")
    def _flag_low_confidence(self) -> "Region":
        if self.confidence == "低" or self.disease == "待判读":
            self.pending_review = True
        return self

    review_notes: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="这一处要人复核的具体点，要能落地。"
        "'请专家确认'不算复核点，'侧光下确认白色块是原绘白粉还是"
        "脱落露地仗'才算",
    )

    # 这个区域是不是 AI 提出的。理论上进卡的都是 AI 提出的，
    # 但保留这个字段，是为了以后人工补标时能一眼看出哪些不是模型给的。
    ai_generated: bool = True

    @field_validator("region_id", mode="before")
    @classmethod
    def _fix_region_id(cls, value: object) -> object:
        """卡片编号写歪了自动纠正，规则见 _normalize_region_id。"""
        return _normalize_region_id(value, "R")


class Recommendation(BaseModel):
    """一条保护 / 数字展示 / 公众传播建议。

    赛题要的是"1-2 条建议，说明建议依据和可能风险"。
    所以一条建议必须由三部分组成，缺一不可：
      action 做什么、basis 凭什么、risk 可能出什么问题。
    risk 设了 min_length=1：允许写"风险较低"，但不允许空着。
    空着就意味着这条建议没经过风险思考，评委一眼能看出来。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    action: str = Field(min_length=4, max_length=60, description="建议做什么")
    basis: str = Field(
        min_length=4,
        max_length=60,
        description="依据：对应哪个区域的哪个病害 + 为什么这么建议",
    )
    risk: str = Field(
        min_length=1,
        max_length=40,
        description="风险：这条建议可能带来的问题，不能留空",
    )
    needs_human_review: bool = Field(
        default=True,
        description="默认 True。不建议任何一条建议绕过人工复核进入材料",
    )


class ReportDraft(BaseModel):
    """C 成稿的输出：卡片上"区域以外"的文字。

    C 不看图片，只负责文字整理，所以它能决定的东西被刻意限制在
    总述、建议、局限三样。区域相关的内容它不许改——
    那是 A 观察官和 B 审查官的职责。这条边界很重要：
    答辩时被问"这句话是谁写的"，必须能一句话答上来。
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=4, max_length=80, description="一句话总述")
    recommendations: list[Recommendation] = Field(
        min_length=1,
        max_length=2,
        description="1-2 条建议，每条都要有依据和风险",
    )
    limitations: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="这张图看不准的地方",
    )

    @model_validator(mode="after")
    def _require_review(self) -> "ReportDraft":
        """模型不能自己决定"这条建议不需要人复核"。

        needs_human_review 是给人工复核流程用的开关，默认 True。
        如果允许模型把它设成 False，等于让被审查的对象决定"我不用被审查"。
        """
        if any(not item.needs_human_review for item in self.recommendations):
            raise ValueError("模型生成的建议必须标记 needs_human_review=true")
        return self


class Provenance(BaseModel):
    """来源与工具披露。

    这一块直接对应评分维度里的"规范合规"：图片及资料来源、模型和工具披露。
    赛题原文要求"完整披露所使用的图片、资料、模型、平台、插件及第三方服务"，
    并且明确"隐瞒第三方 AI 工具使用情况"会被取消成绩——
    所以这几个字段不是形式主义，是保命字段，一个都不能留空。
    """

    image_source: str = Field(
        min_length=1,
        description="图片从哪来：命题方提供 / 数字敦煌 / 自摄 / URL",
    )
    image_license: str = Field(
        min_length=1,
        description="授权情况。没写清楚来源和授权，这张卡就算不合规",
    )
    scene_note: str = Field(
        default="",
        max_length=120,
        description="命题方给的场景说明，原样带过来，别自己改写",
    )
    platform: str = Field(
        default="阿里云百炼（DashScope）",
        description="生成卡片的平台。现场赛只认百炼，这条别写错",
    )
    models: list[str] = Field(
        min_length=1,
        description="实际调用的模型全名，如 ['qwen3.8-max']。"
        "名字写错这件事在'工具披露'这个评分点上很扎眼",
    )
    prompt_version: str = Field(
        min_length=1,
        description="提示词版本：用 git 短哈希或手写版本号。"
        "没有版本号，改坏了就没法回滚，也没法在答辩时说清是哪一版",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="除模型外的工具，如 ['Pillow 12.0（卡片渲染）']。"
        "开源库也要披露",
    )
    generated_at: str = Field(min_length=1, description="ISO8601 时间，便于追责")
    run_id: str = Field(
        min_length=1,
        description="本次运行的唯一编号，和 trace 日志一一对应",
    )


class Attribution(BaseModel):
    """AI 生成 / 人工复核 的归属划分。

    赛题原文（任务描述第 3 条）：
    "明确哪些内容由 AI 生成、哪些需要人工复核，并注明图片、公开资料和 AI 工具来源。"
    所以这三个列表是硬要求，不是可选项：
      - ai_generated：哪些是模型写的
      - human_review_required：哪些必须人看过才能用
      - human_reviewed：人已经看过并确认的（空 = 还没复核）
    """

    ai_generated: list[str] = Field(
        min_length=1,
        description="列举由 AI 生成的内容，例如 '区域 R1-R3 的观察描述与全部建议文字'",
    )
    human_review_required: list[str] = Field(
        min_length=1,
        description="列举必须人工确认的环节，例如 'R2 的病害类型判定'",
    )
    human_reviewed: list[str] = Field(
        default_factory=list,
        description="已经人工确认的部分。空列表表示这张卡还没复核过",
    )
    reviewer: str = Field(default="", description="复核人姓名。留空表示尚未复核")


class Card(BaseModel):
    """一页「AI 壁画观察卡」的完整数据（最终交付物）。

    这是全队唯一的交付口径：渲染器读它出 PDF，报告读它写正文，
    视频读它做字幕。任何人想加字段，先改这里再改文档，别在渲染器里塞私有键。

    extra="forbid" 是故意的：多写一个未声明的字段会直接报错，
    防止某个人偷偷加"给渲染器用"的临时字段，最后只有他自己的机器能跑。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default="1.0",
        description="契约版本号。字段有不兼容改动时加一，方便定位是哪版生成的卡",
    )

    image_id: str = Field(min_length=1, description="图片文件名（不含扩展名）")

    summary: str = Field(
        min_length=4,
        max_length=80,
        description="一句话总述：这张图最值得说的结论。"
        "不要罗列所有病害，卡片正文里已经有区域说明了",
    )

    regions: list[Region] = Field(
        min_length=1,
        max_length=3,
        description="1-3 个最值得关注的区域，按威胁程度从高到低排。"
        "赛题明确只要求 1-3 处，给出 8 处反而不符合交付要求",
    )

    recommendations: list[Recommendation] = Field(
        min_length=1,
        max_length=2,
        description="1-2 条保护 / 数字展示 / 公众传播建议",
    )

    limitations: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="这张图看不准的地方。这是加分项，不是丢人项——"
        "敢写局限，才说明知道 AI 的能力边界在哪",
    )

    provenance: Provenance
    attribution: Attribution

    review_status: Literal["待人工复核", "已人工复核", "已人工复核并修改"] = Field(
        default="待人工复核",
        description="复核状态。标成已复核就必须在 attribution 里写清是谁、改了哪些",
    )

    disclaimer: str = Field(default=DEFAULT_DISCLAIMER)

    @model_validator(mode="after")
    def _check_region_ids(self) -> "Card":
        """编号不能重复。

        重复的话渲染器会把两个区域画成同一个标签，
        卡片上看起来少的不能，其实只是被覆盖了。
        """
        ids = [region.region_id for region in self.regions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"region_id 有重复：{ids}")
        return self

    @model_validator(mode="after")
    def _check_review_claim(self) -> "Card":
        """禁止"空口标注已复核"。

        这条是给答辩准备的：如果卡上写着"已人工复核"，但复核人一栏是空的、
        也没写改了什么，那这个标记就是假的。评委不用查代码，看一眼就能拆穿。
        所以把一致性约束放到数据层，让假标记根本存不下来。
        """
        if self.review_status == "待人工复核":
            return self
        if not self.attribution.reviewer.strip():
            raise ValueError(
                f"review_status='{self.review_status}' 时必须填写 attribution.reviewer"
            )
        if not self.attribution.human_reviewed:
            raise ValueError(
                f"review_status='{self.review_status}' 时 attribution.human_reviewed 不能为空"
            )
        return self


# =============================================================================
# 4. 标注集（内部评测用，不进交付物）
# =============================================================================
# 说明：标注不是为了训练模型，是为了量出"改完提示词到底是变好还是变坏"。
# 详细规则见 docs/annotation.md，这里只把格式钉死。

Severity = Literal["轻", "中", "重"]


class AnnotationRegion(BaseModel):
    """人工标注的一个关注区域。字段和 Region 对齐，只是多了严重程度和排序。"""

    bbox: BoundingBox
    disease: DiseaseName
    # 严重程度允许留空：单张照片看不出轻重，不要逼标注人猜
    severity: Severity | None = None
    note: str = Field(default="", max_length=40, description="一句话备注，给标注人自己看")
    rank: int = Field(ge=1, le=3, description="重要程度排序，1 最重要")


class AnnotationRecord(BaseModel):
    """data/labels.jsonl 里的一行。一行 = 一张图。

    关键约定：
      - 正样本只标 1-3 个最值得关注的区域，不要求标全 8 类。
        理由是交付物只要 1-3 处，标全了既费时间又没法直接对分。
      - 负样本（negative=true）必须单独准备，用来测误报。
        这是整套评测里最容易被忽略、又最影响"问题理解"得分的一类数据。
    """

    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1)
    source: str = Field(min_length=1, description="图片来源，必须能追溯到具体 URL 或'命题方提供'")
    license: str = Field(min_length=1, description="授权说明")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    negative: bool = Field(
        default=False,
        description="True 表示这张图没有需要标记的病害（含纯纹样、完好壁画）",
    )
    # "看不清"和"看清了确实没有"必须分开。
    #
    # 混在一起的后果：拍虚的图被标成负样本，模型说"这有异常"就记成一次误报，
    # 误报率直接失真。所以单独一个字段，评测时这一组只统计拒判，不进召回/误报。
    assessment: Literal["可判读", "无法判断"] = "可判读"
    assessment_note: str = Field(
        default="",
        max_length=200,
        description="无法判读的原因、以及需要补什么证据",
    )
    annotator: str = Field(default="", description="标注人，方便追责和复核")
    regions: list[AnnotationRegion] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _check_consistency(self) -> "AnnotationRecord":
        """负样本不能有框，正样本不能没框，rank 不能重复。"""
        if self.assessment == "无法判断":
            # 无法判读的图既不是负样本，也不该有确诊区域
            if self.negative or self.regions:
                raise ValueError(
                    f"{self.image_id}: assessment=无法判断 时不能标为负样本，也不能给 regions"
                )
            if not self.assessment_note.strip():
                raise ValueError(
                    f"{self.image_id}: assessment=无法判断 时必须写清原因（assessment_note）"
                )
            return self

        if self.negative and self.regions:
            raise ValueError(f"{self.image_id}: negative=true 时 regions 必须为空")
        if not self.negative and not self.regions:
            raise ValueError(f"{self.image_id}: 正样本至少要标 1 个区域")
        ranks = [region.rank for region in self.regions]
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"{self.image_id}: rank 有重复 {ranks}")
        return self


# =============================================================================
# 5. 旧版：A 阶段 v1 的输出（保留兼容，勿在新代码中使用）
# =============================================================================
# 下面这几个类是最早跑通 demo 时用的格式：只有文字位置、没有坐标框，
# 所以它做不出一页卡，只能做一份"病害清单"。
# 现在 workflow.py / result_parser.py 还在用它们，
# 贸然删掉会把已经跑通的链路弄断，所以先留着。
# 迁移路径：A 改成输出 Observation（带框）→ B 裁决 → C 出 Card，
# 三处一起替换，然后删掉这一整段。

DiseaseStatus = Literal["明确发现", "疑似发现", "无法判断", "未发现"]

Confidence = Literal["高", "中", "低", "无法评估"]


class DiseaseFinding(BaseModel):
    """【旧版】一条病害记录，没有坐标，无法在图上标框。"""

    disease: str
    status: DiseaseStatus
    location: list[str] = Field(default_factory=list)
    evidence: str
    confidence: Confidence = "无法评估"
    limitations: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """【旧版】A 阶段 v1 的最终输出。迁移到 Observation 后删除。"""

    image_id: str
    findings: list[DiseaseFinding]
    limitations: list[str] = Field(default_factory=list)
    summary: str
