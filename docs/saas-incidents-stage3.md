# 阶段 3：候选根因验证与离线基线评测

本阶段在阶段 2 的两步只读取证基础上，增加 **候选假设 → 逐项验证 → 归因或拒判**，不增加数据库或 LLM 调用负担。诊断输入仅来自通过演示应用身份验证的 `get_incident_evidence` 返回值，**不能读取离线测试用的 `GROUND_TRUTH`**。

## 行为

- `overview` 只产生候选假设（`status=unverified`）；不会在看见 HTTP 429 时直接断言原因。
- `traffic/auth/export/dependency` 证据到达后，分别标记假设为 `supported`、`rejected` 或 `unknown`，并在诊断结果中保留所使用的证据字段。
- 恰好一个候选被支持且其余候选已能排除时输出 `verdict=supported`；信息不足输出 `UNDETERMINED`；同时支持多个原因则输出 `CONFLICTING_EVIDENCE`，不伪造唯一根因。
- 生成的报告包含：事故编号、时间、根因代码、中文解释、下一步、候选假设及其证据字段、演示快照标记。

## 本地评测

```powershell
.\.venv\Scripts\python.exe -m pytest tests/technical_support/test_diagnosis_stage3.py -q
.\.venv\Scripts\python.exe scripts/evaluate_saas_diagnosis.py
```

评测脚本使用临时 SQLite 数据库，先走 `overview` 再根据症状决定取证维度，最后将规则诊断结果与**离线**标签比较。该脚本**不调用 Qwen**，其准确率只表示固定 8 例合成样本上的规则基线，不能对外宣称 Agent 的端到端准确率或生产效果。

## 下一阶段

采集实际 Qwen 对 F01–F08 的端到端运行：工具轨迹、根因答案、证据是否正确、延迟与 Token 消耗。再考虑更复杂的多轮重新规划；当前的 `next_evidence_view` 依然是有限的两阶段动态取证，不能称为通用自动根因分析。
