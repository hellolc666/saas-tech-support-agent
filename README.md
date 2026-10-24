# SaaS 智能技术支持 Agent

面向虚构“星河协作平台”的本地演示项目：中文知识问答、应用配额与请求日志查询、导出条件诊断、人工确认后创建工单。沿用 LangGraph + Qwen + FastAPI + Streamlit，SaaS 业务数据与会话检查点使用 SQLite。

## 启动

在项目根目录配置 `.env` 的 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`DASHSCOPE_MODEL`，设置 `CUSTOMER_SUPPORT_MODE=qwen`。示例见 `.env.qwen.example`，不要提交真实密钥。

```powershell
# 本机运行
uv sync --frozen
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1
# 或启动 Docker Desktop 后运行容器
 docker compose -f compose.saas.yaml up -d --build
```

打开 http://127.0.0.1:8501 ，API 为 http://127.0.0.1:8080 。端口冲突时设置 `SAAS_API_PORT` / `SAAS_UI_PORT` 再运行 Compose。首次知识检索会下载 BGE embedding 模型，需要网络；模型和索引缓存保存在命名卷中。

```powershell
docker compose -f compose.saas.yaml logs --tail 60 agent_service
docker compose -f compose.saas.yaml stop
docker compose -f compose.saas.yaml start
```

停止容器不会删除数据；不要对需要保留的数据卷使用 `down -v`。`SAAS_ONLY=1` 限制服务注册为 SaaS 入口，`customer-support` 是兼容别名。

## 架构

```mermaid
flowchart LR
 UI[Streamlit] --> API[FastAPI / SSE]
 API --> G[LangGraph + Qwen]
 G --> R[BM25 + Dense + RRF + BGE Reranker]
 R --> C[Chroma + BGE + Markdown]
 G --> T[只读诊断工具]
 T --> DB[(SQLite 演示业务库)]
 G --> H[人工确认 / 取消 / 超时]
 H --> W[幂等工单写入]
 W --> DB
 G <--> CP[(SQLite Checkpoint)]
 G --> L[JSONL Trace]
 UI --> F[本地评分]
 F --> DB
```

知识规则必须带文档来源；动态数据只来自业务工具。导出工具只诊断条件，不生成文件。工单写入先暂停等待确认，默认审批有效期 900 秒（`SAAS_APPROVAL_TTL_SECONDS`）；取消或超时不写入，恢复使用原会话 ID。数据库以租户和幂等键去重，相同键不同内容返回冲突。新的独立提交是新的工单请求。

Trace 位于 `data/logs/saas_trace.jsonl`（容器 `/app/runtime/saas_trace.jsonl`），包含会话 ID、意图、工具、耗时、状态、错误码和检索文档 ID；不记录工具参数或完整提示词。评分按运行 ID 存储到 SQLite，不依赖 LangSmith。模型最终工单成功声明需对应本轮工具返回编号，演示验证码会在最终文本中隐藏。SSE 输出完整校验后的答案，仍提供工具事件。

## 当前检索链路

默认使用 **BM25＋BGE向量检索＋RRF（k=60）＋BAAI/bge-reranker-base**。先融合两路召回，过滤低相关证据，再对前15个候选交叉编码重排，返回前2条。模型实例在进程内复用，CPU并发访问串行化。模型不可用时明确报工具错误，不静默降级为Hybrid。首次使用需下载重排模型约1GB，CPU延迟会高于旧Hybrid。

## 演示与验收

1. 问“API 返回401怎么办？”：检查检索工具与文档引用。
2. 问“APP_DEMO_001 导出5001条失败，验证码 DEMO-ALPHA-4821”：应返回单次5000条限制。
3. 新会话仅问“为什么导出5001条失败”：逐步补应用和验证码，检查多轮参数保持。
4. 要求创建工单，给出应用、验证码、类别与故障摘要：确认前无写入；确认后返回真实编号，继续查询应为“待处理”。另开会话测试取消。
5. 草稿产生后重启 API，保持原会话再确认，验证检查点恢复。等待超过审批有效期应拒绝旧草稿并要求重新提交。
6. 用错误验证码查询配额：应拒绝，不返回其他应用数据。对答案评分后检查 SQLite feedback 表。

合成身份：`APP_DEMO_001 / DEMO-ALPHA-4821`（2.0 编辑者）；`APP_DEMO_002 / DEMO-BETA-7394`（1.0 访客）。这些是公开演示数据，不是生产认证系统。当前不提供真实外部工单通知、生产配额查询、多租户账户管理。会话检查点保存对话，本地部署仍需保护数据库；应用验证码不能替代正式身份鉴权。

## 复跑

```powershell
$env:PYTHONPATH='src'
uv run pytest tests/technical_support tests/service/test_service.py tests/service/test_auth.py -q
uv run python scripts/accept_saas_stage1.py
uv run python scripts/accept_saas_stage2.py --base-url http://127.0.0.1:8080
uv run python scripts/evaluate_saas_ablation.py
```

真实 Qwen 验收会产生模型调用费用。消融脚本会下载并运行 BGE reranker，CPU 运行较慢。实验与测试证据在 `data/evaluation/`，说明见 [阶段验收记录](docs/saas-acceptance.md) 和 [检索消融报告](docs/saas-retrieval.md)。原电商说明保留于 [历史 README](docs/legacy-ecommerce-readme.md)，原 `compose.yaml` 属于历史电商部署。
