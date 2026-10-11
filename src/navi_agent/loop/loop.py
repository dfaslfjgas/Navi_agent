
from langchain_core.messages import AIMessage, BaseMessage

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

    async def run(
        self,
        messages: list[BaseMessage],
        callback: LoopCallback,
    ) -> None:
        """执行 Agent Loop，通过 Callback 通知外部。"""

        # 使用独立历史，避免修改调用方的消息。
        history = list(messages)
        schemas = self._registry.list_schemas()

        for _ in range(self._max_iterations):
            response: AIMessage | None = None

            # 单次 LLM 调用。
            async for output in self._model.stream(history, schemas):
                if isinstance(output, str):
                    await callback(CallbackEvent.TEXT_CHUNK, output)

                elif isinstance(output, AIMessage):
                    response = output

            if response is None:
                raise RuntimeError("Model returned no response")

            # 记录完整的 LLM 响应。
            history.append(response)
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
                history.append(tool_message)
                await callback(CallbackEvent.TOOL_COMPLETED, tool_message)

        raise RuntimeError(
            f"Agent exceeded max_iterations={self._max_iterations}"
        )
