"""Execute intent labels remain optional metadata, separate from shell commands."""

import asyncio
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.backends import LocalShellBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def test_execute_exposes_optional_description_to_model() -> None:
    middleware = FilesystemMiddleware(backend=StateBackend())
    execute = next(tool for tool in middleware.tools if tool.name == "execute")
    schema = execute.tool_call_schema.model_json_schema()

    assert "description" in schema["properties"]
    assert schema["properties"]["description"]["default"] is None
    assert "description" not in schema["required"]


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "description_args",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"description": None}, id="null"),
        pytest.param({"description": ""}, id="empty"),
        pytest.param({"description": "Print a greeting"}, id="label"),
    ],
)
async def test_execute_preserves_description_without_affecting_command(
    tmp_path: Path, *, async_mode: bool, description_args: dict[str, str | None]
) -> None:
    args = {"command": "printf hello", "timeout": 10, **description_args}
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[{"name": "execute", "id": "greeting", "args": args}]),
                AIMessage(content="Done"),
            ]
        )
    )
    agent = create_agent(model, middleware=[FilesystemMiddleware(backend=LocalShellBackend(root_dir=tmp_path))])
    inputs = {"messages": [HumanMessage(content="Print a greeting")]}
    result = await agent.ainvoke(inputs) if async_mode else await asyncio.to_thread(agent.invoke, inputs)

    call = result["messages"][1]
    output = result["messages"][2]
    assert isinstance(call, AIMessage)
    assert call.tool_calls[0]["args"] == args
    assert isinstance(output, ToolMessage)
    assert output.status == "success"
    assert "hello" in output.content
    assert output.artifact == {"exit_code": 0}
