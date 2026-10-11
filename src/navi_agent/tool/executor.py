
import asyncio
from typing import Any, Coroutine

from jsonschema import ValidationError, validate

from src.navi_agent.tool.schema import ToolResult
from .tool import Tool
from .schema import ToolResult

async def execute_tool(
    tool: Tool,
    arguments: dict[str, Any],
    tool_call_id: str,
) -> ToolResult:
    """执行工具，负责参数校验、超时控制和失败重试。"""

    # 校验工具参数，不对参数错误进行重试。
    try:
        validate(instance=arguments, schema=tool.schema.parameters)
    except ValidationError as exc:
        return ToolResult(
            tool_call_id=tool_call_id,
            success=False,
            content="",
            error=f"Invalid tool arguments: {exc.message}",
        )

    metadata = tool.metadata

    # max_retries 表示首次执行失败后的额外尝试次数，负数按 0 处理。
    retries = max(metadata.max_retries, 0)
    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(
                tool.execute(arguments),
                timeout=metadata.timeout,
            )

            return ToolResult(
                tool_call_id=tool_call_id,
                success=True,
                content=result,
            )

        except asyncio.CancelledError:
            # 保留 Harness 的取消语义。
            raise

        except Exception as exc:
            if attempt == retries:
                return ToolResult(
                    tool_call_id=tool_call_id,
                    success=False,
                    content="",
                    error=f"{type(exc).__name__}: {exc}",
                )

    raise RuntimeError("Unexpected tool execution state")

async def tool_execute_max_try(tool_call_id: str,):
    return ToolResult(
        tool_call_id=tool_call_id,
        success=False,
        content="",
        error=f"tool calls max try",
    )
