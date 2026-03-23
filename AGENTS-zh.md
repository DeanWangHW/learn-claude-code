# Repository Guidelines（中文）

## 项目结构与模块组织
- `agents/`：Python 教学实现，覆盖 `s01` 到 `s12`，以及完整版本 `s_full.py`。
- `docs/{en,zh,ja}/`：三语文档，按 session 对应讲解核心机制。
- `skills/`：技能定义与辅助脚本（如 `skills/agent-builder/scripts/init_agent.py`）。
- `web/`：Next.js 16 + TypeScript 可视化学习平台，源码在 `web/src`，静态资源在 `web/public`。
- `.github/workflows/`：CI 配置（Web 构建/类型检查与 Python 测试流程）。

## 构建、测试与开发命令
- `pip install -r requirements.txt`：安装 Python 依赖。
- `cp .env.example .env`：初始化环境变量（至少配置 `ANTHROPIC_API_KEY`、`MODEL_ID`）。
- `python agents/s01_agent_loop.py`：运行第一节最小 Agent 循环。
- `python agents/s12_worktree_task_isolation.py`：运行高级隔离执行示例。
- `python agents/s_full.py`：运行整合版参考实现。
- `cd web && npm ci && npm run dev`：启动 Web 平台（`http://localhost:3000`）。
- `cd web && npm run build`：生产构建（会先执行内容抽取脚本）。
- `cd web && npx tsc --noEmit`：严格类型检查（CI 同步执行）。

## 代码风格与命名约定
- Python 使用 4 空格缩进；函数用 `snake_case`，常量用 `UPPER_SNAKE_CASE`。
- `agents/` 文件命名保持 `sNN_topic.py`（例如 `s08_background_tasks.py`）。
- TypeScript/React 使用严格类型；组件导出名用 PascalCase；组件文件名以 kebab-case 为主。
- Web 代码优先使用 `@/*` 路径别名，减少深层相对路径。

## 测试指南
- 当前仓库未跟踪 `tests/` 目录，Python 变更建议用对应 `agents/sXX_*.py` 做冒烟验证。
- 提交前至少执行：
  - `cd web && npx tsc --noEmit`
  - `cd web && npm run build`
  - 运行你修改到的一个或多个 Python session 脚本。
- 若新增 Python 测试，建议放在 `tests/` 下并使用 `test_*.py` 命名。

## 提交与 PR 规范
- 提交信息建议简短、祈使语气，常见前缀：`feat:`、`fix:`、`fix(scope):`。
- 每个 commit 聚焦单一变更主题，避免混入无关修改。
- PR 应包含：变更摘要、影响路径、验证命令；Web UI 改动附截图。
- 如有关联问题或 PR，请在描述中引用（例如 `(#39)`）。
