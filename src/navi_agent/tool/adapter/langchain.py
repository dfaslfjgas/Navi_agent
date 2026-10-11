
import json
from typing import Any

from langchain_core.messages import ToolMessage

from ..schema import ToolSchema, ToolResult

def to_langchain_tool(schema: ToolSchema) -> dict[str, Any]:
    """将内部工具声明转换为 LangChain Tool Schema。"""
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }


def to_langchain_message(result: ToolResult) -> ToolMessage:
    """将工具执行结果转换为 LangChain ToolMessage。"""
    if not result.success:
        content = result.error or "Tool execution failed"
    elif isinstance(result.content, str):
        content = result.content
    else:
        content = json.dumps(result.content, ensure_ascii=False)

    return ToolMessage(
        content=content,
        tool_call_id=result.tool_call_id,
        status="success" if result.success else "error",
    )
