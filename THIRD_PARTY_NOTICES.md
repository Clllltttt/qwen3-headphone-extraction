# 第三方与数据说明

本项目发布范围为代码、合成数据和实验记录，不包含任何基础模型权重、LoRA adapter、训练检查点或 GGUF。

基础模型为 Qwen/Qwen3-1.7B，由 Qwen 团队提供，使用 Apache License 2.0。请从 https://huggingface.co/Qwen/Qwen3-1.7B 获取模型，并保留其许可证和相关声明。本仓库 LICENSE 使用 Apache License 2.0；不重新授权第三方依赖。

训练依赖 PyTorch、Transformers、PEFT、Accelerate、Datasets 和 bitsandbytes，各依赖按其自身许可证分发，本仓库未附带依赖安装包。

data/ 中为编写素材经程序组合生成的合成耳机评论及标签，不是真实消费者评论，也不是独立人工标注数据。没有声称已经完成独立人工完整复核。原始素材、生成脚本、数据说明及 manifest 一并保留。

scripts/train.py、scripts/evaluate.py 从本项目实验快照整理，增加模型路径参数并调整默认数据路径；历史报告是原实验记录，不是整理后重新训练的结果。reports/ 中本机项目绝对路径用 <LOCAL_PROJECT> 替换，其余实验结果保留。
