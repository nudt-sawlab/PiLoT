# PiLoT 上传规则

## 适用范围

这份规则只用于 `jhvbgg5558/PiLoT` 仓库。

当前分支已经有一套用于保存 DOM/DSM 实验结果的目录习惯：

- `docs/experiments/dom_dsm_prepare/query_16x9_crop_results/renderer_only_512/`
- `docs/experiments/dom_dsm_prepare/query_16x9_crop_results/full_visual_check/`
- `docs/experiments/dom_dsm_prepare/single_full/`

除非当前实验确实需要新的同级目录，否则优先延续这套结构。

## 通常允许上传

- `tools/*.py`
- `pixloc/utils/dom_dsm/*.py`
- `configs/*.yaml`
- `docs/experiments/dom_dsm_prepare/*.md`
- 经过筛选的轻量结果文件：
  - `*.json`
  - `*.txt`
  - `*.md`
  - 用于说明结果的 `*.png`

`outputs/...` 里的小文件不是一律禁止。只要它们满足以下条件，就可以上传：

- 与本次实验直接相关
- 文件体积轻量
- 文件数量有限
- 不是把整个 `outputs/` 大目录原样打包上传

上传时要么逐个文件加入暂存区，要么只加入一个很小的实验子目录，不能直接 `git add outputs`。

## 通常不要上传

- `data_caiwangcun/reference/*.tif`
- `data_caiwangcun/reference/*.tiff`
- `data_caiwangcun/**/*.jpg` 下的大批量原始或生成图像
- `data_caiwangcun/**/*.png` 下的大批量原始或生成图像
- `data_caiwangcun/query/images/...` 下生成的 query 裁剪图
- 权重文件：`*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、`*.safetensors`
- `data_demo/pretrained_model/*`
- `.conda/`
- `.cache/`
- `__pycache__/`
- `*.pyc`
- `runs/`
- `results/`
- `wandb/`

不要把一个体积很大的 `outputs/` 树整体加入版本库。只保留真正需要解释实验或复现实验的小文件。

## 建议检查顺序

1. 看 `git status --short --untracked-files=all`
2. 看大于 `20MB` 的文件列表
3. 对照允许上传项和禁止上传项判断本次改动
4. 如果 `outputs/...` 中有有价值的轻量结果，决定是：
   - 复制到 `docs/experiments/dom_dsm_prepare/...`
   - 还是直接按小范围 `outputs/...` 文件上传
5. 用 `git diff --cached --name-only` 再检查一遍暂存区

## 仓库内现有文档风格

实验说明文档通常包含：

- 实验目的
- 输入
- 运行命令
- 产物清单
- 指标、日志或结果片段
- 解释与结论

结果文件名尽量保持稳定、直观，例如：

- `render_stats_512.json`
- `visual_compare_metrics.json`
- `run_log.json`
- `result_pose.txt`
- `query_render_overlay.png`
- `edge_overlay.png`
- `checkerboard_overlay.png`

## 提交信息参考

- `docs: add query camera geometry diagnostic`
- `docs: add 16x9 query crop validation results`
- `docs: add refined pose visual comparison results`
- `tools: add DOM DSM visual validation script`
- `config: add CaiWangCun DOM DSM 16x9 config`
- `experiment: add single-image DOM DSM validation outputs`
