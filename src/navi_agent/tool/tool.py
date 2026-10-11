
from abc import ABC, abstractmethod
from typing import Any

from .schema import ToolSchema, ToolMetadata


class Tool(ABC):
    """工具抽象接口，由 Harness 统一管理和执行。"""

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """返回工具声明。"""
        ...

    @property
    def metadata(self) -> ToolMetadata:
        """返回工具执行元数据。"""
        return ToolMetadata()

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> Any:
        """执行工具并返回结果。"""
        ...
