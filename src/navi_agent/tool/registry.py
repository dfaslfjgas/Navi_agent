
from .tool import Tool
from .schema import ToolSchema


class ToolRegistry:
    """工具注册中心，负责维护工具实例及其唯一名称。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具，禁止重复名称。"""
        name = tool.schema.name

        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        """注销工具。"""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """根据名称查找工具。"""
        return self._tools.get(name)

    def list_schemas(self) -> list[ToolSchema]:
        """返回所有已注册工具的 Schema。"""
        return [tool.schema for tool in self._tools.values()]
