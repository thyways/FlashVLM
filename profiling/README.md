# Advanced Profiling (Qwen3-VL)

该目录提供一个高级 profiling 脚本，用于分析端到端推理瓶颈，覆盖：

- 视频解码（`decord`）
- `qwen_vl_utils` 视觉输入处理
- processor 打包与上卡
- 视觉编码（Vision Encoder）
- LLM prefill / decode 分离计时
- 各方法（`method/*`）的小组件耗时

## 脚本

- `qwen3_vl_advanced_profile.py`

## 快速使用

```bash
cd "/home/wmk/code/FlashVLM"

python "./profiling/qwen3_vl_advanced_profile.py" \
  --video-path "/path/to/video.mp4" \
  --question "Describe the video in detail." \
  --method fastv \
  --num-frames 160 \
  --max-new-tokens 128 \
  --fastv-r-ratio 0.2
```

## 支持方法

- `none` (baseline)
- `flashvlm`
- `flashvid`
- `vidcom2`
- `fastv`
- `visionzip`
- `holitom`

## 输出

脚本会输出：

1. 终端 bottleneck 表（按总耗时排序）
2. JSON 报告（默认保存在 `profiling/reports/profile_<method>_<timestamp>.json`）

可通过 `--report-path` 指定输出路径。
