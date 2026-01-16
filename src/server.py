import argparse
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--base-url", type=str, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", type=str, help="Model name")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4-turbo")

    if not api_key:
        raise ValueError("API key required. Set OPENAI_API_KEY environment variable or use --api-key")

    skill = AgentSkill(
        id="llm_agent",
        name="LLM Agent",
        description="General-purpose LLM agent with function calling support",
        tags=["llm", "function-calling"],
        examples=[]
    )

    agent_card = AgentCard(
        name="LLM Agent",
        description="OpenAI-compatible LLM agent with automatic function calling support",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(api_key, base_url, model),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print(f"Starting agent with model: {model}")
    print(f"API base URL: {base_url}")

    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
