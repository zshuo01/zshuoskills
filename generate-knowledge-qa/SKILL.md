---
name: generate-knowledge-qa
description: Generate Chinese financial GraphRAG evaluation JSON from atomic knowledge JSON/JSONL, matching each question and answer to at least three relevant supporting knowledge ids.
---

# Generate Knowledge QA

从金融知识生成评测集，重点是问题、答案、支撑 ID 合理对应。直接完成出题与自检，不要求额外的证据映射、复核批次或审计文件。

## 最终格式

最终交付一个 `.json` 文件，根节点为对象，字段与顺序保持不变：

- 顶层：`dataset_name`、`source_file`、`design_principles`、`cases`。
- 每条 case：`query_id`、`category`、`question`、`groundtruth_knowledge_ids`、`groundtruth_answer`、`design_intent`。
- 名称和来源为字符串，设计原则为字符串数组，`cases` 为非空数组；问题按 `Q001`、`Q002` 连续编号。

不添加额外字段，不把中间候选文件或 JSONL 当作最终交付。

## 出题规则

1. 每题至少 **3 个真实、唯一且有实际贡献的知识 ID**。不设目标数量或固定上限；需要多少就选多少，不默认全为3个，也不强求数量分布多样。
2. 围绕自然、明确的业务问题选知识。候选知识只是资料池，可以舍弃无关或重复内容；不能为了凑够3条拼接不相干的问题。若无法自然组成至少3条有效支撑，换题或拒绝。
3. 所选知识应完整支撑问题和答案，每个 ID 都对回答有实际贡献。问题要求“全部材料、所有条件”等完整清单时，核对原文范围，不能只挑部分条目作完整答案。
4. 答案简洁、具体、完整，保留主体、适用条件、否定、例外、时间、数值和单位，不补充外部规则，不以“以相关规定为准”代替答案。计算应使用材料中的事实并核对结果；不设句数硬限制。
5. 问题为自然中文，恰好一个 `？`，无半角 `?`、知识 ID 或“原子知识、知识库、根据文本、根据材料”等措辞。答案使用连续中文表述，不列要点。
6. `category` 为“主题-子主题”，两段各2–6个汉字；`design_intent` 以“测试系统能否召回”或“测试系统能否围绕”开头，简要说明考查的信息。

## 执行

1. 读取用户的知识及来源。识别 ID 字段 `knowledge_id`/`id`/`kid` 和正文 `knowledge_text`/`content`/`text`/`chunk`/`summary`，保留适用条件；重复 ID 或缺正文应先处理。
2. 生成自然问题，同时选择支撑 ID、撰写答案。输出前自检：问题是否答全、每个 ID 是否有用、答案是否忠于原文。证据不足则回看原始知识补齐或换题，不凑数。
3. 保存最终 JSON 并校验。`<skill-dir>` 表示本文件所在目录，执行时替换为实际路径：

   ```bash
   python <skill-dir>/scripts/validate_qa_dataset.py final.json --knowledge-file knowledge.json
   ```

   多个输入可写为 `--knowledge-file json1.json json2.json`。校验器检查结构、ID存在性和至少3条的要求；语义对应由出题时自检负责。

数据量大时可使用 [README.md](README.md) 中的可选批量脚本；只需一轮生成，无额外复核环节。候选池缺少所需证据时，回看原始知识并补入实际读取的材料。关系文件仅用于候选分组，无图结构保留要求。
