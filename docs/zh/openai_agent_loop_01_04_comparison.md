# OpenAI Agents 01-04 `agent_loop` 对比

## 对比范围
- `agents_openai/s01_agent_loop.py:79` 的 `agent_loop`
- `agents_openai/s02_tool_use.py:170` 的 `agent_loop`
- `agents_openai/s03_todo_write.py:246` 的 `agent_loop`
- `agents_openai/s04_subagent.py:229` 的 `agent_loop`
- 另外补充 `agents_openai/s04_subagent.py:198` 的 `run_subagent`，因为它是 `task` 工具的关键差异点。

## 共同点（四个版本都一样的主骨架）
四个版本都遵循同一个 OpenAI 工具调用循环模式：
1. `client.chat.completions.create(...)` 让模型决策（文本输出或工具调用）。
2. 将 `assistant` 消息（含 `tool_calls`）写入 `messages`。
3. 若没有 `tool_calls`，`return` 结束循环。
4. 执行每个工具调用，并将结果作为 `{"role":"tool","tool_call_id":...,"content":...}` 回填到 `messages`。

## 差异矩阵
| 维度 | s01 | s02 | s03 | s04 |
|---|---|---|---|---|
| 可用工具 | 仅 `bash` | `bash/read_file/write_file/edit_file` | s02 + `todo` | `PARENT_TOOLS = CHILD_TOOLS + task` |
| 工具分发方式 | `if tool_call.function.name == "bash"` | `TOOL_HANDLERS` 映射分发 | `TOOL_HANDLERS` 映射分发 | `task` 特判 + 其他走 `TOOL_HANDLERS` |
| 工具入参解析 | 内联 `json.loads` | 内联 `json.loads` | 内联 `json.loads` | 独立 `parse_tool_input` |
| 工具执行异常保护 | 仅 JSON 解析兜底 | 无统一 `try/except`（handler 抛错会中断） | 有统一 `try/except` | 有统一 `try/except` |
| 额外状态机 | 无 | 无 | `rounds_since_todo`，3 轮未更新 todo 注入提醒 | 无 todo 状态机，但有子代理循环上限（30 轮） |
| 子代理能力 | 无 | 无 | 无 | 有：`task -> run_subagent(prompt)`，子上下文隔离 |
| 代码抽象层次 | 最小实现 | 增加 handlers | 增加 handlers + todo 策略 | 增加 `assistant_to_history`/`call_model`/`parse_tool_input` 复用 |

## 逐版演进（重点差异）

### s01: 最小可运行 loop
- 只处理 `bash`，逻辑最直接。
- 适合解释最核心的“请求模型 -> 执行工具 -> 回填结果”闭环。

### s02: 通用工具分发
- 由单一 `bash` 扩展为多工具，并通过 `TOOL_HANDLERS` 做统一分发。
- loop 结构不变，变化点集中在工具集合与 dispatch。

### s03: 在 loop 上加“任务管理策略”
- 在 s02 的基础上增加 `todo` 工具与进度监督。
- 新增 `rounds_since_todo` 计数器；连续 3 轮未触发 `todo` 时注入提醒消息，推动模型更新计划。

### s04: 父子代理双循环
- 父 `agent_loop` 新增 `task` 工具，命中后调用 `run_subagent`。
- `run_subagent` 维护一套独立 `sub_messages`（上下文隔离），只将最终 summary 返回父代理。
- 相比 s01-s03，s04 从“单循环”升级为“父循环 + 子循环”协同。

## 关键结论
- **相同点**：四者的主循环协议完全一致，都是 OpenAI `tool_calls` 回填模式。
- **不同点本质**：差异主要在“工具调度策略与状态管理层”。
- **演进方向**：`s01`（最小闭环） -> `s02`（多工具） -> `s03`（计划监督） -> `s04`（上下文隔离的子代理协作）。
