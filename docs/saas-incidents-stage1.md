# 阶段 1：故障模拟数据与证据读取

本阶段只建立可复现的故障样本及 SQLite 证据读取接口，**尚未将新接口加入 LangGraph 工具清单或 Planner**。这样先验证数据与应用隔离，下一阶段再做动态诊断，避免同时修改多条链路。

- F01/F02：相同的 HTTP 429，但分别是窗口配额耗尽、瞬时速率过高。
- F03/F04/F07：相同的导出失败，但分别是资源权限不足、导出数量超限、版本不支持。
- F05/F06：相同的 HTTP 401，但分别是令牌过期、凭证环境不匹配。
- F08：API 503 / 慢请求，模拟下游依赖劣化。

每个案例都有 `overview`、`traffic`、`auth`、`export` 和 `dependency` 五个可查询维度。工具返回的是**发生故障时的合成快照**，与当前 `api_quotas` 和 `applications` 演示表中的默认快照可以不同；不能把二者当作同一时间窗口的生产事实。

## 当前阶段的 Python 调用方式

```python
from technical_support.incident_scenarios import (
    get_incident_evidence,
    list_incident_scenarios,
)

print(list_incident_scenarios("APP_DEMO_001", "DEMO-ALPHA-4821"))
print(get_incident_evidence("F01", "APP_DEMO_001", "DEMO-ALPHA-4821", "overview"))
print(get_incident_evidence("F01", "APP_DEMO_001", "DEMO-ALPHA-4821", "traffic"))
```

`GROUND_TRUTH` 仅供离线评测。证据接口不返回标签，避免 Agent 直接读取答案。查询必须先通过演示应用验证码校验，跨应用案例与不存在的案例返回统一未找到错误。

## 验收

```powershell
.\.venv\Scripts\python.exe -m pytest tests/technical_support/test_incident_scenarios.py -q
.\.venv\Scripts\python.exe -m pytest tests/technical_support tests/service/test_service.py tests/service/test_auth.py -q
```

下一阶段要做的是：为这套证据增加只读 LangChain 工具，扩展 Planner 的任务类型，并通过“概览 → 证据查询 → 根因判断”的轨迹评测验证诊断能力。不要把本阶段的固定样本直接称作真实生产故障或端到端诊断准确率。
