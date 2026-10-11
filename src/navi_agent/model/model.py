
from collections.abc import AsyncIterator, Sequence
from typing import TypeAlias

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ..tool.schema import ToolSchema
from ..tool.adapter.langchain import to_langchain_tool


ModelOutput: TypeAlias = str | AIMessage


class Model:
    """模型客户端，负责流式输出及完整响应组装。"""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model_name: str,
    ):
        self._llm = ChatOpenAI(
            api_key=SecretStr(api_key),
            base_url=api_url,
            model=model_name,
            streaming=True,
        )

    async def stream(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[ModelOutput]:
        """实时输出文本，并在结束时返回完整 AIMessage。"""

        llm = self._llm

        if tools:
            schemas = [to_langchain_tool(tool) for tool in tools]
            llm = llm.bind_tools(schemas)

        response: AIMessageChunk | None = None

        async for chunk in llm.astream(list(messages)):
            response = chunk if response is None else response + chunk

            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content

        if response is None:
            return

        yield AIMessage(
            content=response.content,
            tool_calls=response.tool_calls,
            invalid_tool_calls=response.invalid_tool_calls,
            response_metadata=response.response_metadata,
            usage_metadata=response.usage_metadata,
        )
