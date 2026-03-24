#!/usr/bin/env python3
"""OpenAI Agent 观测模块统一出口。

Notes
-----
业务脚本只需要导入 ``create_observer`` 并调用统一生命周期方法，不需要关心：

- Langfuse SDK 初始化细节
- 凭证缺失/网络失败时的降级策略
- 不同脚本之间打点格式的一致性
"""

from .langfuse_observer import BaseObserver, create_observer

__all__ = ["BaseObserver", "create_observer"]
