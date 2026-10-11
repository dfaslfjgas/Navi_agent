
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSchema:
    """提供给模型的工具声明。"""

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
        }
    )


@dataclass(frozen=True)
class ToolMetadata:
    """提供给 Harness 的工具执行元数据。"""

    timeout: float = 30.0
    max_retries: int = 0


@dataclass
class ToolResult:
    """工具执行结果。"""

    tool_call_id: str
    success: bool
    content: str
    error: str | None = None
