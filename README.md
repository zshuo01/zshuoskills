# Zshuo Skills

这是一个存放自定义 Agent 技能（Skills）的仓库。每个技能都是一个独立的包，包含执行特定复杂任务的说明（`SKILL.md`）、脚本和测试用例。

## 技能列表 (Skills List)

| 技能名称 (Skill Name) | 一句话讲解 (Description) | 调用示例 (Usage Examples) |
| :--- | :--- | :--- |
| **generate-knowledge-qa** | 从原子知识 JSON/JSONL 数据中，通过本地主题聚类、大模型改写合并等步骤，自动生成符合 GraphRAG 本地查询格式的评估数据集。 | **1. 生成候选评估数据骨架**:<br>`python generate-knowledge-qa/scripts/generate_qa_groundtruth.py 原子知识.json -o qa_candidates.json --count 30`<br><br>**2. 导出 LLM 改写 prompts**:<br>`python generate-knowledge-qa/scripts/rewrite_and_merge.py qa_candidates_review.json --export-prompts -o rewrite_prompts.jsonl`<br><br>**3. 合并改写结果**:<br>`python generate-knowledge-qa/scripts/rewrite_and_merge.py qa_candidates_review.json --merge rewritten_results.jsonl -o graphrag_local_eval_final.json`<br><br>**4. 校验数据集格式**:<br>`python generate-knowledge-qa/scripts/validate_qa_dataset.py graphrag_local_eval_final.json` |

## 如何使用本仓库

1. 确保您的环境已安装 Python 并具备 Git 工具。
2. 将此仓库克隆或集成到您的 Agent 执行环境中。
3. 遵循各个技能目录下的 `SKILL.md` 指引或运行相应的脚本。
