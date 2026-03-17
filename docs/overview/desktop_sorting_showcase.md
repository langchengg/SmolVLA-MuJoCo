# Desktop Sorting Showcase

这个部分是仓库的默认主线，也是最适合直接放到 GitHub 首页的视频化 demo。

## 目标

通过一个自包含的 MuJoCo 桌面场景，演示：

- 文本指令驱动的抓取与放置
- 多物体排序
- paraphrase 泛化
- 视觉扰动鲁棒性
- zero-shot vs finetuned mock policy 对比

## 默认运行方式

```bash
python3 -m pip install -e .
python3 scripts/run_showcase_demo.py --output artifacts/showcase
```

运行完成后会生成：

- `artifacts/showcase/episodes/desktop_sorting_eval_log.jsonl`
- `artifacts/showcase/summary/desktop_sorting_eval_summary.csv`
- `artifacts/showcase/summary/summary.md`
- `artifacts/showcase/figures/*.png`
- `artifacts/showcase/videos/*.mp4`
- `artifacts/showcase/videos/hero_showcase.gif`

## 模块拆分

`src/portfolio_vla/scene.py`

- 负责构建 MuJoCo 桌面场景
- 定义两侧 tray、后方 bin、桌面物体和轻量机械臂
- 提供离屏渲染和简单抓放动作轨迹

`src/portfolio_vla/tasks.py`

- 读取 `templates/desktop_sorting_prompts.yaml`
- 将自然语言模板解析成对象选择规则与放置规则

`src/portfolio_vla/policy.py`

- 提供 `zero_shot` 和 `finetuned` 两种 mock policy
- 注入可复现的 success / failure 模式

`src/portfolio_vla/runner.py`

- 执行 episode
- 写 JSONL / CSV / Markdown
- 生成图表和视频

## 你可以怎么扩展

### 1. 增加 prompt

在 [templates/desktop_sorting_prompts.yaml](../../templates/desktop_sorting_prompts.yaml) 中新增一条规则即可。

建议优先加：

- 新的 paraphrase 指令
- 新的 compositional rule
- 新的 visual perturbation case

### 2. 增加物体

在 [configs/desktop_sorting_showcase.yaml](../../configs/desktop_sorting_showcase.yaml) 的 `task_family.objects` 中新增对象名，并在 `src/portfolio_vla/tasks.py` 的 `size_map` 中定义尺寸。

### 3. 增加新的 mock policy

你可以新增：

- `quantized_mock`
- `rl_refined_mock`
- `moveit_baseline_mock`

这样可以在不引入真实大模型训练的前提下，先把 GitHub 结果图和对比故事搭起来。

## 适合放在 README 顶部的视频

最推荐的视频组合：

- 一个 nominal success clip
- 一个 paraphrase clip
- 一个 perturbation clip

仓库已经把这三类片段合成成：

- `artifacts/showcase/videos/hero_showcase.mp4`
- `artifacts/showcase/videos/hero_showcase.gif`

## 实话实说怎么写

这个 showcase 最加分的写法不是“我做了具身智能大模型部署”，而是：

- simulation-first
- reproducible
- instruction-conditioned
- episode-level logging
- perturbation analysis

也就是说，要把项目包装成“一个完整的研究/作品集工程”，而不是“一个假装真机部署的 demo”。
