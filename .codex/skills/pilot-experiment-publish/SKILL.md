---
name: pilot-experiment-publish
description: 在 `jhvbgg5558/PiLoT` 仓库中整理并发布一次本地实验的 GitHub 提交。适用于实验刚跑完、需要收集本次新增或修改的脚本、配置、实验说明文档和轻量结果文件，排除数据集、权重、环境目录和大体积输出，然后提交到当前功能分支、推送，并在 GitHub 上复核上传内容的场景。
---

# PiLoT 实验结果上传

用于在 PiLoT 仓库中完成一次实验后的收尾上传。
先检查工作区，再精确选择要提交的文件，避免把大数据、权重和无关输出一并推上去，最后再到 GitHub 上核对结果。

## 工作流程

开始暂存前先阅读 [references/pilot-upload-rules.md](references/pilot-upload-rules.md)。其中整理了这个仓库专用的允许上传项、禁止上传项、`docs/experiments/dom_dsm_prepare/` 现有目录习惯，以及提交信息命名建议。

## 先检查

在仓库根目录先执行：

```powershell
git branch --show-current
git status --short --untracked-files=all
Get-ChildItem -Recurse -File | Where-Object { $_.Length -gt 20MB } | Sort-Object FullName | ForEach-Object { '{0}`t{1:N2} MB' -f $_.FullName, ($_.Length/1MB) }
```

如果实验产物在 `outputs/` 下，先判断应该如何保留：

- 如果只是少量、明确、轻量的实验结果，可以直接保留在一个小范围的 `outputs/...` 子目录里并精确上传
- 如果这些结果需要和实验说明文档一起长期保留，优先复制到 `docs/experiments/dom_dsm_prepare/...`

不要直接把整个 `outputs/` 目录加入暂存区。

## 精确选择文件

优先使用显式 `git add`，只添加本次实验直接相关的文件。常见类别：

- `tools/*.py`
- `pixloc/utils/dom_dsm/*.py`
- `configs/*.yaml`
- `docs/experiments/dom_dsm_prepare/*.md`
- 轻量结果文件，例如 `*.json`、`*.txt`、`*.md`，以及少量说明用途的 `*.png`

如果某个结果当前只存在于 `outputs/...` 中，先决定它是否应该：

- 复制到 `docs/experiments/dom_dsm_prepare/...` 后再上传，或
- 直接以小范围 `outputs/...` 文件的形式上传

不要把生成出来的 query 图像上传到 `data_caiwangcun/query/images/`。

## 检查暂存区

执行：

```powershell
git diff --cached --name-only
$files = git diff --cached --name-only; if ($files) { $files | ForEach-Object { $p = Join-Path (Get-Location) $_; if (Test-Path $p) { '{0}`t{1:N2} KB' -f $_, ((Get-Item $p).Length/1KB) } } }
```

如果发现不该提交的路径被加进来了，立刻移出暂存区：

```powershell
git restore --staged <path>
```

除非用户明确要求，否则以下内容视为禁止上传：

- 原始影像、DOM、DSM、裁剪后的大批量 query 图像
- 模型权重
- `.conda/`、`.cache/`、`__pycache__/`、`*.pyc`
- 整个 `runs/`、`results/`、`wandb/` 之类的运行目录
- 没有经过筛选的整块 `outputs/` 子树

## 提交并推送

根据本次改动的主类型选择提交信息，例如：

- `docs: add <实验主题> results`
- `tools: add <实验主题> utility`
- `config: add <实验主题> config`
- `experiment: add <实验主题> outputs`

然后执行：

```powershell
git commit -m "<message>"
git push origin <current-branch>
git status
```

除非用户明确要求，否则不要改写已有提交。

## 在 GitHub 上复核

推送后确认：

- GitHub 上目标分支的最新 commit 已更新
- 本次上传的脚本、配置、文档和结果文件都能在目标分支中看到
- 没有误传大数据、权重、环境目录或无关的大输出

优先使用 GitHub 插件读取最新 commit 和代表性文件，完成最终核对。

## 默认策略

默认只提交“最小但足够”的文件集合。
如果本次实验只改了一个文档或一个配置，就只提交那个文件。
如果同一次实验同时新增了脚本和一组轻量结果，可以合并在一个提交里，但前提是它们确实属于同一实验闭环。
