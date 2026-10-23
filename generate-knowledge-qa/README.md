# Generate Knowledge QA

规则与最终 JSON 格式见 [SKILL.md](SKILL.md)。通常直接读取知识、生成问题/答案/至少3个支撑ID，再运行校验器即可。ID数量按实际需要确定，无固定目标或上限；不足3条有效支撑就换题，不凑数。

## 可选批量工具

从仓库根目录运行；已安装的 skill 请替换脚本目录：

```bash
python generate-knowledge-qa/scripts/generate_qa_groundtruth.py knowledge.json -o candidates.json --count 30
python generate-knowledge-qa/scripts/rewrite_and_merge.py candidates.json --export-prompts -o prompts.jsonl
```

使用 LLM 处理每个提示词一次，将六字段用例或带 `query_id` 的 `reject_reason` 存为 `results.jsonl`，然后合并：

```bash
python generate-knowledge-qa/scripts/rewrite_and_merge.py candidates.json --merge results.jsonl -o final.json
```

候选只包含 `query_id`、分类提示和完整支撑文本，不预填最终 ID。默认每个候选池最多12条材料，可用 `--pool-size N` 调整，或用 `--pool-size 0` 包含整个相关分组；这是读取批次大小，不是最终证据数量要求。生成器支持多个输入文件及原有的可选 `--relations` 分组。

合并采用模型选出的真实 ID 子集，自动检查原有四个顶层字段、六个用例字段、ID存在性、唯一性和至少3条。不需要片段映射、哈希或第二轮复核。语义是否对应在生成时自检；程序不声称能自动证明答案正确。

无效项打印原因，可选 `--rejected-output rejected.json` 保存报告。缺失/重复结果会报错；没有有效用例时不写最终文件。失败时已有输出保持原样。

直接生成最终文件后，可检查一个或多个原始知识文件：

```bash
python generate-knowledge-qa/scripts/validate_qa_dataset.py final.json --knowledge-file knowledge.json
python -B -m unittest discover -s generate-knowledge-qa/tests -v
```
