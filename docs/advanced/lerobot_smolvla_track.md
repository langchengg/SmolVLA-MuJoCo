# Advanced Track: SmolVLA + LeRobot + LIBERO

这个仓库的默认主线是自包含 showcase，但它也保留了你后续继续“做深”的入口。

## 什么时候走 advanced 路线

当你已经有下面这些基础成果后，再上 advanced 路线最合适：

- GitHub README 完整
- showcase 视频已生成
- mock benchmark 图表已生成
- 你已经能清楚讲明白项目定位

这时候再接入真实 `SmolVLA + LIBERO + LeRobot`，会让项目从“展示型 portfolio”升级成“复现 + 分析 + 扩展”的研究型 portfolio。

## 推荐顺序

### Step 1. 安装仓库和 LeRobot

```bash
python3 -m pip install -e .
bash scripts/bootstrap_lerobot.sh
```

如果在 notebook / cloud 环境里遇到 `egl_probe` 构建问题，可以使用：

```bash
LEROBOT_SKIP_EGL_PROBE=1 bash scripts/bootstrap_lerobot.sh
```

### Step 2. 跑最小 fine-tuning

```bash
HF_USER=your_hf_username \
POLICY_REPO_ID=your_hf_username/libero-smolvla-demo \
MUJOCO_GL=egl \
bash scripts/train_libero_smolvla.sh
```

说明：

- `HF_USER` 是你的 Hugging Face 用户名
- `POLICY_REPO_ID` 是你想保存 checkpoint 的 HF repo id
- `MUJOCO_GL=egl` 适合 headless / cloud 环境

### Step 3. 跑评估

```bash
POLICY_PATH=your_hf_username/libero-smolvla-demo \
TASKS=libero_10 \
MUJOCO_GL=egl \
bash scripts/eval_libero_smolvla.sh
```

### Step 4. 跟当前仓库的结果系统对接

建议把 advanced 路线的结果也收敛成当前仓库同样的“可讲述”格式：

- benchmark CSV
- figure PNG
- summary Markdown
- 失败案例视频

这样你整个 GitHub 仓库就会形成一条自然演进线：

1. quick demo path 先保证完整度
2. advanced track 再补真实性和研究深度

## Hugging Face Login 什么时候需要

不上传任何东西时：

- 可以不 login

需要上传 checkpoint / 访问私有资源时：

- 必须先 `huggingface-cli login`

## 项目表述建议

最稳的写法不是“我做了真实部署”，而是：

- reproduced and extended an open-source VLA stack
- evaluated policy behavior in MuJoCo / LIBERO
- compared zero-shot and fine-tuned policies
- used a self-contained showcase to package results for GitHub and resume presentation

## 你后续还能继续加什么

- real SmolVLA checkpoints
- RL fine-tuning in MuJoCo
- MoveIt planning baseline
- zero-shot vs fine-tuned vs quantized comparison
- domain randomization experiments
- demo transfer story from benchmark task to showcase task
