import json
import re
import os
import logging
from openai import OpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.messenger = Messenger()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.context_to_history = {}
        self.context_to_tools = {}  # Cache tools for each context

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        # print("Input text:", input_text)
        user_query, tools = self._parse_input(input_text)

        # Get context_id and initialize history for this context
        context_id = message.context_id

        # Initialize conversation history for new context with system prompt
        if context_id not in self.context_to_history:
            self.context_to_history[context_id] = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the provided tools to help answer the user's query. Call the necessary functions to gather information before providing your final answer."
                }
            ]

        # Cache tools for this context (first call will have tools, subsequent calls may not)
        if tools is not None:
            self.context_to_tools[context_id] = tools
        else:
            # Use cached tools if available
            tools = self.context_to_tools.get(context_id, None)

        history = self.context_to_history[context_id]
        history.append({
            "role": "user",
            "content": user_query
        })

        await updater.update_status(
            TaskState.working, new_agent_text_message("Processing...")
        )

        try:
            response = self._call_llm(tools, context_id)
            result = self._format_response(response, context_id)
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=result))],
                name="Response",
            )
        except Exception as e:
            logger.error(f"Error in agent.run: {e}")
            raise

    def _parse_input(self, text: str) -> tuple[str, list | None]:
        """
        Parse input text to extract query and tools.

        Strategy:
        1. Try to extract tools and query from structured format
        2. If parsing fails, use entire text as query (fallback to tau2-style)

        This allows the agent to work with both:
        - Structured formats (ComplexFuncBench initial call)
        - Plain text (tau2-style, or ComplexFuncBench subsequent calls)
        """
        tools = None
        query = text
        parsed_successfully = False

        # Try parsing "User query:" format (first call)
        if "User query:" in text:
            query_match = re.search(r'User query:\s*(.+?)$', text, re.DOTALL)
            if query_match:
                query = query_match.group(1).strip()
                parsed_successfully = True

            tools_start_match = re.search(r'tools:\s*\n\s*(\[)', text, re.DOTALL)
            if tools_start_match:
                start_pos = tools_start_match.start(1)
                bracket_count = 0
                end_pos = start_pos
                for i in range(start_pos, len(text)):
                    if text[i] == '[':
                        bracket_count += 1
                    elif text[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_pos = i + 1
                            break

                if end_pos > start_pos:
                    try:
                        tools_json = text[start_pos:end_pos]
                        tools_data = json.loads(tools_json)
                        tools = self._convert_to_openai_format(tools_data)
                    except Exception:
                        pass

        # Try parsing pure JSON format
        if not parsed_successfully:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    if "tools" in data:
                        tools = self._convert_to_openai_format(data["tools"])
                    if "query" in data:
                        query = data["query"]
                        parsed_successfully = True
            except json.JSONDecodeError:
                pass

        # Fallback: if no structured parsing succeeded, use entire text as query
        # This handles tau2-style plain text messages like "Tool result: ..."
        if not parsed_successfully:
            query = text
            # Try to extract tools from anywhere in the text
            tools_match = re.search(r'(?:following tools|Available tools):\s*(\[.+?\])', text, re.DOTALL)
            if tools_match:
                try:
                    tools_data = json.loads(tools_match.group(1))
                    tools = self._convert_to_openai_format(tools_data)
                except:
                    pass

        return query, tools

    def _convert_to_openai_format(self, tools_data: list) -> list:
        openai_tools = []
        for tool in tools_data:
            if tool.get("type") == "function" and "function" in tool:
                openai_tools.append(tool)
            else:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {})
                    }
                })
        return openai_tools

    def _call_llm(self, tools: list | None, context_id: str):
        history = self.context_to_history[context_id]
        messages = history.copy()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0
        }

        if tools:
            kwargs["tools"] = tools

        try:
            print("Tools provided to LLM:", json.dumps(tools, indent=2))
            print("messages sent to LLM:", json.dumps(messages, indent=2))
            response = self.client.chat.completions.create(**kwargs)
            print("response from LLM:", response.choices[0].message)
            print("="*50)
            return response
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def _format_response(self, response, context_id: str) -> str:
        message = response.choices[0].message
        history = self.context_to_history[context_id]

        if message.tool_calls:
            history.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Always use OpenAI standard format
            tool_calls_output = []
            for tc in message.tool_calls:
                tool_calls_output.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    }
                })
            return json.dumps({"tool_calls": tool_calls_output}, ensure_ascii=False)

        elif message.content:
            history.append({
                "role": "assistant",
                "content": message.content
            })

            if self._has_previous_tool_calls(context_id):
                return json.dumps({"response": message.content}, ensure_ascii=False)
            else:
                return message.content

        return ""

    def _has_previous_tool_calls(self, context_id: str) -> bool:
        history = self.context_to_history[context_id]
        for msg in history:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                return True
        return False
