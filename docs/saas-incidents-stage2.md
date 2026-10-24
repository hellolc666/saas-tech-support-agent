# 阶段 2：基于证据的逐步故障诊断（演示数据）

在已有阶段 1 模拟器基础上增加 `list_incident_scenarios` 与 `get_incident_evidence` 两个只读工具。只有通过演示应用 ID / 验证码核验才能访问对应故障。故障案例是固定历史合成数据，不代表当前生产运行状态。

## 执行路径

1. 用户指定合成故障 ID（例如 F01）、应用 ID 和演示验证码；缺少身份则追问。
2. 已有 Qwen Planner 规划 `get_incident_evidence` 任务；如果遗漏明确给出的案例 ID，Middleware 为其补充任务。
3. Agent 先调用 `get_incident_evidence(view="overview")`。返回现象、错误码、状态码，不包含根因答案。
4. 根据上一步观测，选择第二类证据：HTTP 429 → traffic、401 → auth、导出 → export、503 → dependency。模型可以自主调用；若提前输出最终答案，任务中间件会补查未完成的必要证据。
5. `diagnosis.py` 用客观证据生成可复现的基线判定，传递给 Agent 作为已验证数据。无法满足判定条件时返回 `UNDETERMINED`，不能凭空确定根因。
6. 最终回答需要陈述症状、已核对证据、根因及下一步，并说明是历史模拟快照。

这里有**受证据约束的两阶段路由**，并非自主选择任意深度/长度的完整动态规划器。后续阶段可进一步支持多分支假设、追加多工具调查与可解释诊断报告。

## 示例

在 Qwen 模式新建会话，输入：

`请诊断 APP_DEMO_001 的合成故障 F01，演示验证码 DEMO-ALPHA-4821。`

预期：日志出现对 `get_incident_evidence` 的 overview 和 traffic 两次调用，根因 `QUOTA_EXHAUSTED`。同样流程 F02 根因应为 `BURST_RATE_LIMIT`。

## 验收

```powershell
.\.venv\Scripts\python.exe -m pytest tests/technical_support/test_diagnosis_stage2.py -q
.\.venv\Scripts\python.exe -m pytest tests/technical_support tests/service/test_service.py tests/service/test_auth.py -q
```

此阶段单元测试使用受控演示数据。真实 Qwen 轨迹、根因准确率与端到端延迟须另行测量，不能仅凭单元测试通过宣称真实模型的根因诊断准确率。
