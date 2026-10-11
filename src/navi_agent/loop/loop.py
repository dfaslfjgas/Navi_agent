
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from ..model.model import Model
from ..tool.registry import ToolRegistry
from ..tool.executor import execute_tool
from ..tool.schema import ToolResult
from ..tool.adapter.langchain import to_langchain_message
from .callback import CallbackEvent, LoopCallback


class AgentLoop:
    """Agent 执行循环，负责模型调用与工具调度。"""

    def __init__(
        self,
        model: Model,
        registry: ToolRegistry,
        max_iterations: int = 25,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        self._model = model
        self._registry = registry
        self._max_iterations = max_iterations

    async def loop(
        self,
        messages: list[BaseMessage],
        callback: LoopCallback,
    ) -> None:
        """执行 Agent Loop，通过 Callback 通知外部。"""

        # 使用独立历史，避免修改调用方的消息。
        chat_context = list(messages)
        schemas = self._registry.list_schemas()
        finalize_at = max(0, self._max_iterations - 3)

        for loop_cnt in range(self._max_iterations):
            response: AIMessage | None = None

            allow_tools = loop_cnt < finalize_at
            if loop_cnt == finalize_at:
                chat_context.append(SystemMessage(
                    content=(
                        "警告：当前任务已达到工具调用轮次限制，"
                        "后续不允许继续调用任何工具。"
                        "请基于现有对话历史和工具执行结果完成收尾，"
                        "直接给出最终回复。"
                        "如果任务尚未完成，请如实说明已完成的部分和剩余工作，"
                        "不要编造未执行的操作或结果。"
                    )
                ))


            await callback(CallbackEvent.PRELLM, chat_context)
            # 单次 LLM 调用。
            async for output in self._model.stream(chat_context, schemas if allow_tools else None):
                if isinstance(output, str):
                    await callback(CallbackEvent.TEXT_CHUNK, output)

                elif isinstance(output, AIMessage):
                    response = output

            if response is None:
                raise RuntimeError("Model returned no response")

            # 记录完整的 LLM 响应。
            chat_context.append(response)
            await callback(CallbackEvent.LLM_RESPONSE, response)

            if response.invalid_tool_calls:
                raise ValueError(
                    f"Invalid tool calls: {response.invalid_tool_calls}"
                )

            # 没有工具调用，结束 Agent Loop。
            if not response.tool_calls:
                return

            # 查找并执行工具，回填工具结果。
            for call in response.tool_calls:
                # 不允许调用工具的时候 loop 强制返回错误
                if not allow_tools:
                    result = ToolResult(
                        tool_call_id=call["id"],
                        success=False,
                        content="",
                        error="Tool execution is disabled: iteration limit reached",
                    )
                else:
                    tool = self._registry.get(call["name"])

                    if tool is None:
                        result = ToolResult(
                            tool_call_id=call["id"],
                            success=False,
                            content="",
                            error=f"Tool not found: {call['name']}",
                        )
                    else:
                        result = await execute_tool(
                            tool=tool,
                            arguments=call["args"],
                            tool_call_id=call["id"],
                        )

                # 记录并通知工具执行结果。
                tool_message = to_langchain_message(result)
                chat_context.append(tool_message)
                await callback(CallbackEvent.TOOL_COMPLETED, tool_message)

        # TODO: 后续修改为LoopFinalizationError
        raise RuntimeError(
            "Agent failed to finalize: model requested tools "
            "after tool execution was disabled"
        )