# 阶段 4：真实 Qwen 端到端诊断评测

本阶段不改 Agent 逻辑，直接通过已有 FastAPI `/saas-support/stream` SSE 接口发起 **真实 Qwen** 诊断，保存 F01–F08 的案例级结果。需要正常启动 `CUSTOMER_SUPPORT_MODE=qwen` 服务，评测会产生 API 调用费用。

## 命令

```powershell
.\.venv\Scripts\python.exe -m pytest tests/technical_support/test_diagnosis_live_eval.py -q
# 先跑两例：
.\.venv\Scripts\python.exe scripts/evaluate_saas_diagnosis_live.py --cases F01,F02
# 确认成功后再跑8例：
.\.venv\Scripts\python.exe scripts/evaluate_saas_diagnosis_live.py
# 如果使用18505前端/18085后端容器，则加 --base-url http://127.0.0.1:18085
```

默认输出 `data/evaluation/saas-diagnosis-agent-live.json`。包含每个案例的工具名称、证据维度、最终回答（演示验证码/API Key 脱敏）、耗时、正确/错误/人工复核状态。默认脚本为顺序运行以控制开销，运行过程中不得关闭 API 服务。

## 评分语义和局限

- “自动判正确”要求最终回答有单独的 `根因代码：CODE` 行、英文根因与离线标签一致、真实工具观察满足先 overview 再分维度取证。
- 输出没有该格式会计为 `manual_review`，不是自动判错；因此 `strict_accuracy_lower_bound` 是当前自动确认的下界。
- 仅统计 SSE 显示的工具调用与消息；未收到 Token usage 时记录 `null`，不可当作零。
- 8 个案例都是同一批固定合成历史快照，**不是独立测试集，更不能声称生产故障准确率**。至少重复运行，并加入未见过的故障组合与缺失证据测试，才能评估泛化能力。
- `GROUND_TRUTH` 仅由评测脚本读取，不写入用户问题及工具响应。
- 如本地日志或 Qwen SSE 出现异常，报告列为 `error`，不要将其当作正常完成。
