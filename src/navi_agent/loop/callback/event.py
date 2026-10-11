from enum import StrEnum


class CallbackEvent(StrEnum):
    """Agent Loop 执行事件。"""

    TEXT_CHUNK = "text_chunk"
    LLM_RESPONSE = "llm_response"
    TOOL_COMPLETED = "tool_completed"
    PRELLM = "prellm"