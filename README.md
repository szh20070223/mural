# mural
各位这个是项目的框架
有些我ignore 掉了
完整的项目：
data: 图片（没有传）
    问问ai,搜集一下相关壁画图片，重点观察以下病害：
                        1. 裂隙
                        2. 起甲
                        3. 颜料层脱落
                        4. 地仗层脱落
                        5. 粉化
                        6. 褪色
                        7. 水渍
                        
output:是输出（没有传）
prompt：是提示词（这个在仓库里面）
src:调用过程（这个也在仓库）

.env:api（
DASHSCOPE_API_KEY=这个是密钥
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.8-max
）没有传

---

## 文档索引

动手写代码前先看这两份，字段以它们为准：

- [docs/card-contract.md](docs/card-contract.md) — **一页观察卡的数据契约**。
  最终交付物长什么样、每个字段多少字、谁能改什么，都在这里。
- [docs/annotation.md](docs/annotation.md) — **标注规范**。
  标什么、不标什么、框怎么画、验收标准。

配套的示例和工具：

- [docs/card.example.json](docs/card.example.json) — 一张填满字数上限的示例卡，
  做版式直接拿它排。能排下它，真实数据就都能排下。
- [data/labels.example.jsonl](data/labels.example.jsonl) — 标注格式示例。
- `tools/check_contract.py` — 契约自检，改完东西跑一下：

  ```bash
  python tools/check_contract.py
  python tools/check_contract.py --labels data/labels.jsonl
  ```

## 一句话说清现在的状态

- **能跑**：单张图 → A 观察（≤5 个带框候选）→ B 审查（逐条裁决、砍误报）→ C 成稿（建议+风险）→ **一页 A4 PDF**
- **还没做**：标注集、回归评测、客户端硬化（大图缩放/网络重试）、演示前端
- **别训模型**：现场赛只认百炼平台的模型能力

一次运行的产物：

| 目录 | 内容 |
|---|---|
| `output/raw/` | 模型原始回复（每次尝试一份，出问题先看这里） |
| `output/observations/` | A 的候选区域 |
| `output/reviews/` | B 的逐条裁决（含每条的理由和复核点） |
| `output/cards/` | `card.json` + 一页观察卡 `card.pdf`（同名 `.png` 是预览图） |
| `output/trace/` | 运行日志，一行一个事件 |

一条命令出成品卡：

```bash
python main.py --source "命题方提供的比赛材料" --license "仅限本次比赛使用"
```

跑一次：

```bash
python main.py data/images/test_mural1.png
```

两个实测数字（qwen3.8-max，同一张 1123×756 的壁画）：

| | 耗时 | 输出 token | 候选数 |
|---|---|---|---|
| 思考链开着 | 109 秒 | 4955 | 1 个 |
| 思考链关掉 | 13 秒 | 675 | 4 个 |

所以默认关闭思考链（`QWEN_ENABLE_THINKING`）。这个任务要的是"把看到的说清楚"，
不需要长链推理；想太久反而容易收敛到一个结论上。
