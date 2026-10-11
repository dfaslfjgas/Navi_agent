from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias


LoopCallback: TypeAlias = Callable[
    [str, Any],
    Awaitable[Any],
]