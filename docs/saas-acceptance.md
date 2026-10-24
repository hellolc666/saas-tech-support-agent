# SaaS 阶段验收记录（2026-10-23）

## 阶段 1：通过

新增 check_export_eligibility，读取 SQLite 中应用版本和 export_access 的角色、资源访问权限。
APP_DEMO_001 为 2.0、编辑者；APP_DEMO_002 为 1.0、访客。权限与版本不依赖模型猜测。
仅做导出条件诊断，不生成文件。演示业务对象统一使用 APP_DEMO，不引入 PRJ100。
覆盖 1/5000/5001 条、非法数量、验证码不匹配、版本不支持、访客和资源访问受限、多轮补参。
业务工具中的 can_access 是固定演示权限，不是生产身份系统。

验证：SaaS + FastAPI 接口与鉴权回归 64 passed；知识库实测 33 章节、65 用例，
53 条可回答问题 Hit@1=98.11%、Hit@3=100%、MRR@3=98.74%；6 条无关问题无命中率100%。
另6条缺失事实用例要求人工评估回答，检索命中不代表回答正确。P95=16.58ms（本次热启动检索）。

复跑：`.\.venv\Scripts\python.exe scripts/accept_saas_stage1.py`

## 阶段 2：真实 Qwen 小规模验收通过

使用项目 .env 的有效配置，模型 qwen3.7-plus。之前的401鉴权前置问题已解决。
saas-support 是默认入口，customer-support 保留兼容。

2026-10-23 验收证据：
- 401知识问答：Qwen 实际调用 search_technical_docs，返回可核对的文档来源。
- 5001条导出：实际调用 check_export_eligibility 和文档检索，数据库结果为2.0、编辑者、单次5000条、EXPORT_LIMIT_EXCEEDED。
- 三轮补参：先询问5001条导出失败，再补应用ID与演示验证码，最终工具仍收到正确数量并给出诊断。
- 身份拒绝：跨应用验证码查询配额返回 APP_NOT_FOUND_OR_VERIFICATION_FAILED，没有泄露配额。
- 版本与权限：APP_DEMO_002返回1.0、访客、版本不支持和无导出权限，模型如实解释。
- SSE：真实Qwen检索问答返回流事件、来源与[DONE]，没有error事件。
- 故障注入：仅在独立验收进程令配额工具抛异常；中间件将其转为 TOOL_EXECUTION_FAILED，真实Qwen告知失败，没有编造配额，内部异常详情不传给模型。

7项真实模型检查通过。另已通过浏览器在Streamlit提交5001条导出问题，页面展示真实工具调用与超限结论。
以上是小规模冒烟验收，不等同于阶段5的综合Agent评测或线上可靠性保证。
SaaS、FastAPI接口和鉴权回归65 passed；静态检查和git diff --check通过。
工具异常处理保留LangGraph的interrupt控制流；已有工单确认/取消回归继续通过。

复跑（会产生真实模型调用费用）：先启动qwen服务，再运行：
`.\.venv\Scripts\python.exe scripts/accept_saas_stage2.py --base-url http://127.0.0.1:8080`
该脚本覆盖前6项HTTP/SSE检查；故障注入的自动化回归在test_knowledge_agent.py中。
详细逐项证据已导出到本次聊天的stage2-qwen-results.json，网页截图为saas-qwen-acceptance.png。

本轮HTTP与网页使用隔离的SaaS服务与SQLite验收库，未启动其他无关Agent；
测试地址为FastAPI 18082、Streamlit 18503。常规启动脚本仍使用8080/8501。
当前网页进程启动早于工具异常补丁；补丁已用最新代码单元测试与独立真实模型故障注入验证，常规重启后生效。

### 用户复核

在网页输入“为什么 APP_DEMO_001 导出5001条失败？演示验证码 DEMO-ALPHA-4821”。
预期：看到check_export_eligibility工具调用、5000条限制和知识来源；没有声称已生成文件。
另开新对话，输入“为什么导出5001条失败”，按追问补应用和验证码，核对上下文保持。

以上为阶段1–2的历史验收记录；后续阶段结果见下方最终验收。


## 阶段3–6最终技术验收（2026-10-23）

开发与技术自验已完成，等待用户最终验收。最新验收入口为 http://127.0.0.1:18505 ，API 为 http://127.0.0.1:18085 。这两个入口运行 Docker Compose 最新构建，不是此前18503的旧本机进程。

### 阶段3：工单闭环

- 人工确认前不写库；取消不写库；默认900秒审批超时拒绝旧草稿。
- SQLite Checkpoint 关闭连接后重新构图恢复，approve/reject两条路径自动化通过。
- 真实 Qwen 同样执行断开检查点连接后重新加载：确认新增恰好1条，返回编号可查询为待处理；取消新增0条。证据：`data/evaluation/saas-hitl-qwen.json`。
- 并发重复提交、幂等冲突、应用身份隔离、失败不伪造成功均有回归。
- 工单成功声明必须匹配本轮工具结果；不接受历史编号或夹带虚构编号。

### 阶段4：检索消融

同一65条SaaS用例比较 lexical、Dense、Hybrid、BM25、RRF、CrossEncoder。Hybrid Recall@5=100%、MRR@5=98.74%；重排MRR@5=100%，但P95约5.5秒。保留Hybrid默认路径，其他策略保留实验实现。详见 `docs/saas-retrieval.md`。这些指标是检索质量，不能替代最终回答质量。

### 阶段5：可观测性与质量

- 93项SaaS及服务自动化回归全部通过，证据 `data/evaluation/saas-tests.xml`。包括多轮参数、边界数据、身份拒绝、工具故障、确认取消、重启、超时、并发幂等和输出结果校验。这93项含受控模型与数据库测试，不是93次真实LLM测评。
- 容器内6项真实Qwen冒烟全部通过：401文档、5001导出、多轮补参、错误身份、版本权限、SSE。证据 `data/evaluation/saas-docker-qwen.json`。此前独立真实Qwen工具异常注入也通过。
- Trace记录会话、模型/工具耗时、状态、意图、错误码和检索文档ID，不收集完整参数/提示词。
- 网页五星评分实测写入SQLite feedback表（验收时1条），Trace实测29个model/tool事件。
- 最终文本对工单成功结果做确定性核验，并隐藏演示验证码；其他知识解释依赖检索证据与提示约束，并不宣称实现了通用事实判定器。

### 阶段6：部署与交付

Docker Desktop 29.8.0已启动。`compose.saas.yaml`实际构建运行成功，API健康检查通过，Streamlit网页实测问答与评分成功。SaaS只使用SQLite；运行数据、模型、向量库分别持久化到命名卷。README已更新为SaaS主入口，原电商README保留到docs。

实际验收项目名为 `saas-acceptance`，复用该环境：

```powershell
$env:SAAS_API_PORT='18085'
$env:SAAS_UI_PORT='18505'
docker compose -p saas-acceptance -f compose.saas.yaml up -d
docker compose -p saas-acceptance -f compose.saas.yaml ps
```

若docker命令未在PATH中，本机CLI位于 `C:\Users\dafd\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe`。后续正常新部署默认8080/8501。不要在验收数据需要保留时删除命名卷。

### 用户最终验收清单

1. 打开18505，新会话问“API返回401怎么办”：检查检索与引用。
2. 输入“APP_DEMO_001 导出5001条失败，演示验证码 DEMO-ALPHA-4821”：检查2.0、编辑者、5000条上限和文档来源。
3. 新会话只问“为什么导出5001条失败”，按追问补参，核对上下文。
4. 要求创建工单，提供应用、验证码、类别、摘要；先检查待确认内容，再确认并查询编号状态。另开新会话测试取消。
5. 使用错误验证码查询配额，应拒绝且不泄露数据；给答案评分应成功保存。

当前为合成数据本地演示：不生成真实导出文件，不连接生产配额/外部工单系统。文档缺失事实和完整自然语言答案仍需人工复核。已有5条第三方依赖弃用警告，不影响本次93项测试通过。最终静态检查与git diff --check通过；未创建提交，保留原有用户修改。


## 默认检索链路更新（2026-10-23）

按用户最新要求，默认在线路径已改为 BM25＋向量检索＋RRF＋BGE Reranker。上文保留Hybrid的结论为历史性能取舍记录，当前不再适用。重排模型为BAAI/bge-reranker-base，RRF k=60，重排候选最多15条，在线返回前2条。CPU延迟代价已接受。

本次切换后回归为95项通过；容器内真实Qwen知识问答和导出诊断2项均通过，工具引用包含 `retrieval_pipeline=bm25+dense+rrf+bge-reranker`、RRF融合分数及真实重排分数。证据：`data/evaluation/saas-reranker-live.json`。容器18505已更新为该默认链路。
