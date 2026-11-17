# 基于 Qwen3 的耳机评论信息提取与 QLoRA 微调

基于 **Qwen3-1.7B**，从中文耳机评论中提取评价方面、情感、原文依据和复购意愿。项目包含合成数据构建、NF4 QLoRA 训练、验证集早停选择、结构化生成与评估记录。

**本仓库不包含任何模型权重、LoRA adapter、训练检查点或 GGUF 文件。** 使用者需要自行下载基础模型并运行训练。数据是素材组合生成的合成评论，成绩不能代表真实用户评论的泛化表现。

## 任务示例

输入：`声音清楚，戴久了耳朵疼，之后不会再买这款。`

预期结构：

```json
{"评价":[{"方面":"音质","情感":"正面","依据":"声音清楚"},{"方面":"舒适度","情感":"负面","依据":"戴久了耳朵疼"}],"复购意愿":"不愿意"}
```

方面限定为音质、舒适度、续航、连接、物流、客服、价格；情感为正面、负面、中性、正负混合。复购意愿需要明确表述，不能从满意程度直接推断。完整标注规范见 `data/qwen3-dataset-v2/task_prompt.txt`。

## 数据和目录

- `scripts/train.py`：从基础模型新建 QLoRA，按验证损失选择最佳检查点。
- `scripts/evaluate.py`：使用相同生成设置评估基础模型或本地 adapter。
- `scripts/scoring.py`：结构、方面与情感、复购意愿及连续原文依据检查。
- `data/qwen3-dataset-v1/`：500 条训练、100 条验证、100 条原测试数据，含生成脚本、素材和来源说明。
- `data/qwen3-dataset-v2/`：原 500 条加 200 条针对性合成数据，共 700 条训练数据；验证集不变。
- `data/qwen3-aspect-baseline/`、`data/qwen3-aspect-training/`：数据生成时用于排除旧示例的历史素材。
- `reports/`：已完成的历史实验结果、协议、逐条预测和对比报告。

数据按素材划分后分别组合；无完全相同评论不代表没有语义或模板相似性。自动格式和原文检查不能代替独立人工语义复核。

## 环境

原实验使用 Python 3.11、PyTorch 2.9.1+cu128、Transformers 4.57.3、PEFT 0.17.1、bitsandbytes 0.48.2。训练和评估脚本要求 CUDA GPU；原环境固定依赖见 `requirements.txt`，其他系统/CUDA 组合需要自行验证兼容性。

```bash
python -m venv .venv
# 激活虚拟环境后安装 CUDA 12.8 版本 PyTorch
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

从官方仓库 https://huggingface.co/Qwen/Qwen3-1.7B 下载完整基础模型到 `models/Qwen3-1.7B/`。脚本使用本地文件和离线加载，不会自动下载模型。

## 训练和评估

在仓库根目录执行；命令中的模型目录也可以替换为本机已有目录。

```bash
# 基础模型验证基线
python scripts/evaluate.py --model models/Qwen3-1.7B --output runs/base-validation

# 700 条数据，最多 5 轮，连续 2 轮验证损失不改善则早停
python scripts/train.py --model models/Qwen3-1.7B --data data/qwen3-dataset-v2 --expected-train-rows 700 --epochs 5 --patience 2 --output runs/qlora-v3

# 从磁盘加载最佳 adapter 做生成评测
python scripts/evaluate.py --model models/Qwen3-1.7B --adapter runs/qlora-v3/adapter --output runs/v3-validation

# 原测试集回归检查；不应作为新的独立最终测试
python scripts/evaluate.py --model models/Qwen3-1.7B --adapter runs/qlora-v3/adapter --split test --output runs/v3-regression
```

输出目录必须是新目录，避免覆盖实验。训练使用 batch=1、梯度累积 4、学习率 1e-4、余弦衰减、5% 预热、最大长度 1024；LoRA r=8、alpha=16，作用于 q_proj/v_proj，共 1,605,632 个可训练参数。采用 NF4 双重量化，仅对 assistant 答案和 EOS 计算损失，关闭 thinking。

## 已有实验结果

以下来自历史实验记录；整理代码后没有重新执行 GPU 训练。

| 原 100 条合成测试集的回归评测 | 基础模型 | v1：500 条 | v3：700 条+早停 |
| --- | ---: | ---: | ---: |
| 方面与情感 micro F1 | 44.6% | 93.2% | 94.5% |
| 复购意愿准确率 | 72.0% | 100.0% | 100.0% |
| 字段与标签合规率 | 87.0% | 98.0% | 100.0% |
| 整条标签全部正确 | 11.0% | 79.0% | 85.0% |
| 标签、格式与原文依据检查全部通过 | 8.0% | 73.0% | 81.0% |

v3 实际训练 4 轮，选择第 2 轮 checkpoint-350；100 条验证集方面与情感 F1 为 97.4%，验证损失 0.019894，记录的峰值 GPU 分配内存约 3.91 GB（不是总显存需求）。

**评估边界：** 原测试集的错误分析已经指导新增 200 条训练数据，因此 v3 结果属于开发后的回归评测；验证集用于选择检查点且多次查看。二者均不能作为独立真实评论成绩。依据检查只验证连续原文，不保证语义支持。v2/v3 的学习率轨迹也不同，不能把差异全部归因于早停。

## 复现与许可

已附冻结数据，可直接用于训练。`data/qwen3-dataset-v1/build_dataset.py` 可重建 v1；v2 生成与分词审计脚本默认读取仓库下 `models/Qwen3-1.7B` 的本地分词器。manifest 保存数据 SHA-256。

本仓库使用 Apache License 2.0。基础模型由 Qwen 团队提供，其许可和第三方依赖说明见 `THIRD_PARTY_NOTICES.md`。本地实验还完成过合并模型与 Ollama 导出；此次代码发布聚焦数据、训练和评估流程，未包含部署权重。
