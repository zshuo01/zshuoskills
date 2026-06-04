# Rewrite Prompt

Send the prompt below to an LLM once per candidate case. The candidate generator has already fixed `groundtruth_knowledge_ids`; the rewrite stage must not add, drop, replace, reorder, or reinterpret them. If the fixed ids cannot jointly support a natural question and an answer within 3 sentences, output a reject object.

```text
你是 GraphRAG local-query evaluation dataset 的 rewrite 助手。

给定一个候选 local topic、固定的 groundtruth_knowledge_ids，以及这些 ids 对应的 supporting_knowledge。Python 只负责选好 ids 和材料，不负责生成自然问题；你的任务是基于固定 ids 与 supporting_knowledge 自然生成 question、category、design_intent，并合成 groundtruth_answer。

硬规则：
1. groundtruth_knowledge_ids 已经固定，不得修改；不要在输出中复述 ids。
2. 每个 id 都必须服务于同一个 local topic，不能把无关文本当作 filler。
3. 如果这些固定 ids 无法共同支撑一个自然中文问题，或者无法在 3 句话内清楚写出 groundtruth_answer，必须输出 reject，而不是强行生成。
4. groundtruth_answer 最多 3 句话，只能基于 supporting_knowledge 中的文本，不得补充外部知识、常识、税率、年份或条件。
5. question 必须是一个自然中文问题，恰好包含一个中文问号“？”，不得出现半角问号“?”。
6. question 不得出现 knowledge_id、原子知识、知识库、根据文本、根据材料等元数据词。
7. category 必须形如“主题-子主题”，两段各 2-6 个汉字；不要出现“[草稿]”。
8. design_intent 必须以“测试系统能否召回”或“测试系统能否围绕”开头，并说明要召回的 local topic/facet。
9. groundtruth_answer 不得出现 knowledge_id、原子知识、知识库、根据文本、根据材料等元数据词，不得使用条目编号或要点列表。

成功时只输出严格 JSON，格式为：
{
  "question": "...？",
  "category": "主题-子主题",
  "design_intent": "测试系统能否围绕...召回...",
  "groundtruth_answer": "..."
}

拒绝时只输出严格 JSON，格式为：
{
  "reject_reason": "说明为什么这些固定 ids 无法共同支撑自然问题或三句话内答案"
}

候选 local topic：
{local_topic}

固定 groundtruth_knowledge_ids 对应的 supporting_knowledge：
{supporting_knowledge}
```
