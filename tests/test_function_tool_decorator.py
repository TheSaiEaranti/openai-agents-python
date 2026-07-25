from __future__ import annotations

import asyncio
import functools
import inspect
import json
import sys
from types import ModuleType
from typing import Any, cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from agents import UserError, function_tool
from agents.run_context import RunContextWrapper
from agents.tool_context import ToolContext


class DummyContext:
    def __init__(self):
        self.data = "something"


def ctx_wrapper() -> ToolContext[DummyContext]:
    return ToolContext(
        context=DummyContext(), tool_name="dummy", tool_call_id="1", tool_arguments=""
    )


class WrappedPayload(BaseModel):
    value: int


class CallableWrapper:
    def __init__(self, wrapped: Any) -> None:
        self.wrapped = wrapped
        functools.update_wrapper(self, wrapped)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped(*args, **kwargs)


@CallableWrapper
def wrapped_payload_handler(payload: WrappedPayload) -> str:
    return str(payload.value)


@function_tool
def sync_no_context_no_args() -> str:
    return "test_1"


@pytest.mark.asyncio
async def test_sync_no_context_no_args_invocation():
    tool = sync_no_context_no_args
    output = await tool.on_invoke_tool(ctx_wrapper(), "")
    assert output == "test_1"


@function_tool
def sync_no_context_with_args(a: int, b: int) -> int:
    return a + b


@pytest.mark.asyncio
async def test_sync_no_context_with_args_invocation():
    tool = sync_no_context_with_args
    input_data = {"a": 5, "b": 7}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert int(output) == 12


@function_tool
def sync_with_context(ctx: ToolContext[DummyContext], name: str) -> str:
    return f"{name}_{ctx.context.data}"


@pytest.mark.asyncio
async def test_sync_with_context_invocation():
    tool = sync_with_context
    input_data = {"name": "Alice"}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "Alice_something"


@function_tool
async def async_no_context(a: int, b: int) -> int:
    await asyncio.sleep(0)  # Just to illustrate async
    return a * b


@pytest.mark.asyncio
async def test_async_no_context_invocation():
    tool = async_no_context
    input_data = {"a": 3, "b": 4}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert int(output) == 12


@function_tool
async def async_with_context(ctx: ToolContext[DummyContext], prefix: str, num: int) -> str:
    await asyncio.sleep(0)
    return f"{prefix}-{num}-{ctx.context.data}"


@pytest.mark.asyncio
async def test_async_with_context_invocation():
    tool = async_with_context
    input_data = {"prefix": "Value", "num": 42}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "Value-42-something"


@function_tool(name_override="my_custom_tool", description_override="custom desc")
def sync_no_context_override() -> str:
    return "override_result"


@pytest.mark.asyncio
async def test_sync_no_context_override_invocation():
    tool = sync_no_context_override
    assert tool.name == "my_custom_tool"
    assert tool.description == "custom desc"
    output = await tool.on_invoke_tool(ctx_wrapper(), "")
    assert output == "override_result"


@function_tool(failure_error_function=None)
def will_fail_on_bad_json(x: int) -> int:
    return x * 2  # pragma: no cover


@pytest.mark.asyncio
async def test_error_on_invalid_json():
    tool = will_fail_on_bad_json
    # Passing an invalid JSON string
    with pytest.raises(Exception) as exc_info:
        await tool.on_invoke_tool(ctx_wrapper(), "{not valid json}")
    assert "Invalid JSON input for tool" in str(exc_info.value)


def sync_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
    return f"error_{error.__class__.__name__}"


@function_tool(failure_error_function=sync_error_handler)
def will_not_fail_on_bad_json(x: int) -> int:
    return x * 2  # pragma: no cover


@pytest.mark.asyncio
async def test_no_error_on_invalid_json():
    tool = will_not_fail_on_bad_json
    # Passing an invalid JSON string
    result = await tool.on_invoke_tool(ctx_wrapper(), "{not valid json}")
    assert result == "error_ModelBehaviorError"


def async_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
    return f"error_{error.__class__.__name__}"


@function_tool(failure_error_function=sync_error_handler)
def will_not_fail_on_bad_json_async(x: int) -> int:
    return x * 2  # pragma: no cover


@pytest.mark.asyncio
async def test_no_error_on_invalid_json_async():
    tool = will_not_fail_on_bad_json_async
    result = await tool.on_invoke_tool(ctx_wrapper(), "{not valid json}")
    assert result == "error_ModelBehaviorError"


@function_tool(defer_loading=True)
def deferred_lookup(customer_id: str) -> str:
    return customer_id


def test_function_tool_defer_loading():
    assert deferred_lookup.defer_loading is True


@function_tool(strict_mode=False)
def optional_param_function(a: int, b: int | None = None) -> str:
    if b is None:
        return f"{a}_no_b"
    return f"{a}_{b}"


@pytest.mark.asyncio
async def test_non_strict_mode_function():
    tool = optional_param_function

    assert tool.strict_json_schema is False, "strict_json_schema should be False"

    assert tool.params_json_schema.get("required") == ["a"], "required should only be a"

    input_data = {"a": 5}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "5_no_b"

    input_data = {"a": 5, "b": 10}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "5_10"


@function_tool(strict_mode=False)
def all_optional_params_function(
    x: int = 42,
    y: str = "hello",
    z: int | None = None,
) -> str:
    if z is None:
        return f"{x}_{y}_no_z"
    return f"{x}_{y}_{z}"


@pytest.mark.asyncio
async def test_all_optional_params_function():
    tool = all_optional_params_function

    assert tool.strict_json_schema is False, "strict_json_schema should be False"

    assert tool.params_json_schema.get("required") is None, "required should be empty"

    input_data: dict[str, Any] = {}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "42_hello_no_z"

    input_data = {"x": 10, "y": "world"}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "10_world_no_z"

    input_data = {"x": 10, "y": "world", "z": 99}
    output = await tool.on_invoke_tool(ctx_wrapper(), json.dumps(input_data))
    assert output == "10_world_99"


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city.

    Args:
        city: The city to get the weather for.
    """
    return f"The weather in {city} is sunny."


@pytest.mark.asyncio
async def test_extract_descriptions_from_docstring():
    """Ensure that we extract function and param descriptions from docstrings."""

    tool = get_weather
    assert tool.description == "Get the weather for a given city."
    params_json_schema = tool.params_json_schema
    assert params_json_schema == snapshot(
        {
            "type": "object",
            "properties": {
                "city": {
                    "description": "The city to get the weather for.",
                    "title": "City",
                    "type": "string",
                }
            },
            "title": "get_weather_args",
            "required": ["city"],
            "additionalProperties": False,
        }
    )


@function_tool(
    timeout=1.25,
    timeout_behavior="raise_exception",
    timeout_error_function=sync_error_handler,
)
async def timeout_configured_tool() -> str:
    return "ok"


def test_decorator_timeout_configuration_is_applied() -> None:
    assert timeout_configured_tool.timeout_seconds == 1.25
    assert timeout_configured_tool.timeout_behavior == "raise_exception"
    assert timeout_configured_tool.timeout_error_function is sync_error_handler


@pytest.mark.asyncio
async def test_async_callable_object_works_as_bare_function_tool() -> None:
    class AsyncCallable:
        """Double a value.

        Args:
            value: The value to double.
        """

        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, value: int) -> int:
            self.calls += 1
            await asyncio.sleep(0)
            return value * 2

    handler = AsyncCallable()
    tool = function_tool(handler)

    assert tool.name == "AsyncCallable"
    assert tool.description == "Double a value."
    assert tool.params_json_schema["properties"]["value"] == {
        "description": "The value to double.",
        "title": "Value",
        "type": "integer",
    }
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"value": 4}') == 8
    assert handler.calls == 1


@pytest.mark.asyncio
async def test_callable_object_uses_call_docstring_when_class_docstring_missing() -> None:
    class AsyncCallable:
        async def __call__(self, value: int) -> int:
            """Double a value.

            Args:
                value: The value to double.
            """
            return value * 2

    tool = function_tool(AsyncCallable())

    assert tool.description == "Double a value."
    assert tool.params_json_schema["properties"]["value"] == {
        "description": "The value to double.",
        "title": "Value",
        "type": "integer",
    }
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"value": 4}') == 8


@pytest.mark.asyncio
async def test_async_callable_object_works_with_configured_function_tool() -> None:
    class AsyncCallable:
        async def __call__(self, value: int) -> int:
            return value + 1

    configured_function_tool = function_tool(
        name_override="increment",
        description_override="Increment a value.",
        timeout=1,
    )
    tool = configured_function_tool(AsyncCallable())

    assert tool.name == "increment"
    assert tool.description == "Increment a value."
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"value": 4}') == 5


@pytest.mark.asyncio
async def test_sync_callable_object_awaits_awaitable_result_once() -> None:
    class AwaitableReturningCallable:
        def __init__(self) -> None:
            self.calls = 0
            self.awaits = 0

        def __call__(self, value: int) -> Any:
            self.calls += 1

            async def result() -> int:
                self.awaits += 1
                return value * 3

            return result()

    handler = AwaitableReturningCallable()
    tool = function_tool(handler)

    assert await tool.on_invoke_tool(ctx_wrapper(), '{"value": 4}') == 12
    assert handler.calls == 1
    assert handler.awaits == 1


@pytest.mark.asyncio
async def test_callable_wrapper_preserves_published_annotations() -> None:
    tool = function_tool(wrapped_payload_handler)

    assert tool.params_json_schema["properties"]["payload"] == {"$ref": "#/$defs/WrappedPayload"}
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"payload": {"value": 4}}') == "4"


@pytest.mark.asyncio
async def test_nested_callable_wrapper_resolves_call_annotations() -> None:
    class Inner:
        class Payload(BaseModel):
            value: int

        async def __call__(self, payload: Payload) -> int:
            return payload.value

    tool = function_tool(CallableWrapper(Inner()))

    assert tool.params_json_schema["properties"]["payload"] == {"$ref": "#/$defs/Payload"}
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"payload": {"value": 4}}') == 4


@pytest.mark.parametrize("publish_annotations", [False, True])
def test_callable_wrapper_honors_custom_signature(publish_annotations: bool) -> None:
    class CustomSignatureWrapper:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped
            functools.update_wrapper(self, wrapped)
            self.__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        "value",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    )
                ],
                return_annotation="WrappedPayload",
            )
            if publish_annotations:
                self.__annotations__ = {"value": str, "return": WrappedPayload}

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.wrapped(*args, **kwargs)

    def wrapped(value: int) -> Any:
        return WrappedPayload(value=int(value))

    handler = CustomSignatureWrapper(wrapped)
    tool = function_tool(handler, allowed_callers=["programmatic"])

    assert tool.params_json_schema["properties"]["value"]["type"] == "string"
    assert tool.output_json_schema is not None
    assert tool.output_json_schema["title"] == "WrappedPayload"


def test_custom_signature_resolves_wrapper_module_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decorator_module_name = "tests._function_tool_decorator_module"
    target_module_name = "tests._function_tool_target_module"
    decorator_module = ModuleType(decorator_module_name)
    target_module = ModuleType(target_module_name)

    class DecoratorPayload(BaseModel):
        value: int

    class Wrapper:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped
            functools.update_wrapper(self, wrapped)
            self.__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        "payload",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation="DecoratorPayload",
                    )
                ]
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.wrapped(*args, **kwargs)

    DecoratorPayload.__module__ = decorator_module_name
    Wrapper.__module__ = decorator_module_name
    decorator_namespace = cast(Any, decorator_module)
    decorator_namespace.DecoratorPayload = DecoratorPayload
    decorator_namespace.Wrapper = Wrapper
    exec("def target(payload: int) -> int:\n    return payload", target_module.__dict__)
    monkeypatch.setitem(sys.modules, decorator_module_name, decorator_module)
    monkeypatch.setitem(sys.modules, target_module_name, target_module)

    target_namespace = cast(Any, target_module)
    tool = function_tool(Wrapper(target_namespace.target))

    assert tool.params_json_schema["properties"]["payload"] == {"$ref": "#/$defs/DecoratorPayload"}


def test_function_wrapper_custom_signature_resolves_wrapper_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decorator_module_name = "tests._function_tool_function_decorator_module"
    target_module_name = "tests._function_tool_function_target_module"
    decorator_module = ModuleType(decorator_module_name)
    target_module = ModuleType(target_module_name)
    monkeypatch.setitem(sys.modules, decorator_module_name, decorator_module)
    monkeypatch.setitem(sys.modules, target_module_name, target_module)

    exec(
        """
import functools
import inspect
from pydantic import BaseModel

class DecoratorPayload(BaseModel):
    value: int

def decorate(wrapped):
    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    wrapper.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter(
                "payload",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation="DecoratorPayload",
            )
        ]
    )
    return wrapper
""",
        decorator_module.__dict__,
    )
    exec("def target(payload: int) -> int:\n    return payload", target_module.__dict__)

    decorator_namespace = cast(Any, decorator_module)
    target_namespace = cast(Any, target_module)
    tool = function_tool(decorator_namespace.decorate(target_namespace.target))

    assert tool.params_json_schema["properties"]["payload"] == {"$ref": "#/$defs/DecoratorPayload"}


@pytest.mark.asyncio
async def test_partial_async_callable_resolves_postponed_annotations() -> None:
    async def handler(prefix: str, payload: WrappedPayload) -> str:
        return f"{prefix}:{payload.value}"

    partial_handler = functools.partial(handler, "value")
    tool = function_tool(partial_handler)

    assert tool.params_json_schema["properties"]["payload"] == {"$ref": "#/$defs/WrappedPayload"}
    assert "prefix" not in tool.params_json_schema["properties"]
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"payload": {"value": 4}}') == "value:4"


@pytest.mark.asyncio
async def test_partial_uses_wrapped_callable_documentation() -> None:
    async def handler(prefix: str, value: int) -> str:
        """Combine a prefix and value.

        Args:
            prefix: The bound prefix.
            value: The value to combine.
        """
        return f"{prefix}:{value}"

    tool = function_tool(functools.partial(handler, "item"))

    assert tool.name == "handler"
    assert tool.description == "Combine a prefix and value."
    assert "prefix" not in tool.params_json_schema["properties"]
    assert tool.params_json_schema["properties"]["value"] == {
        "description": "The value to combine.",
        "title": "Value",
        "type": "integer",
    }
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"value": 4}') == "item:4"


@pytest.mark.asyncio
async def test_partial_keyword_bound_context_uses_live_context() -> None:
    def handler(ctx: ToolContext[DummyContext], value: int) -> str:
        return f"{ctx.context.data}:{value}"

    bound_context = ctx_wrapper()
    bound_context.context.data = "bound"
    live_context = ctx_wrapper()
    live_context.context.data = "live"
    tool = function_tool(functools.partial(handler, ctx=bound_context))

    assert list(tool.params_json_schema["properties"]) == ["value"]
    assert await tool.on_invoke_tool(live_context, '{"value": 4}') == "live:4"


@pytest.mark.asyncio
async def test_keyword_only_context_is_injected_by_keyword() -> None:
    async def handler(*, ctx: ToolContext[DummyContext], value: int) -> str:
        return f"{ctx.context.data}:{value}"

    live_context = ctx_wrapper()
    live_context.context.data = "live"
    tool = function_tool(handler)

    assert list(tool.params_json_schema["properties"]) == ["value"]
    assert await tool.on_invoke_tool(live_context, '{"value": 4}') == "live:4"


@pytest.mark.asyncio
async def test_variadic_positional_context_is_injected_as_one_argument() -> None:
    async def handler(*ctx: ToolContext[DummyContext]) -> str:
        assert len(ctx) == 1
        return ctx[0].context.data

    live_context = ctx_wrapper()
    live_context.context.data = "live"
    tool = function_tool(handler)

    assert tool.params_json_schema["properties"] == {}
    assert await tool.on_invoke_tool(live_context, "{}") == "live"


@pytest.mark.asyncio
async def test_partialmethod_callable_resolves_class_scoped_annotations() -> None:
    class Handler:
        class Payload(BaseModel):
            value: int

        async def handle(self, prefix: str, payload: Payload) -> str:
            return f"{prefix}:{payload.value}"

        __call__ = functools.partialmethod(handle, "value")

    tool = function_tool(Handler())

    assert tool.params_json_schema["properties"]["payload"] == {"$ref": "#/$defs/Payload"}
    assert "prefix" not in tool.params_json_schema["properties"]
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"payload": {"value": 4}}') == "value:4"


@pytest.mark.asyncio
async def test_partialmethod_async_callable_supports_timeout() -> None:
    class Handler:
        async def handle(self, multiplier: int, value: int) -> int:
            return multiplier * value

        __call__ = functools.partialmethod(handle, 2)

    tool = function_tool(Handler(), timeout=1)

    assert tool.timeout_seconds == 1
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"value": 4}') == 8


def test_partial_cannot_positionally_bind_context() -> None:
    def handler(ctx: ToolContext[DummyContext], value: int) -> str:
        return f"{ctx.context.data}:{value}"

    with pytest.raises(UserError, match="positionally bind"):
        function_tool(functools.partial(handler, ctx_wrapper()))


def test_partial_tools_preserve_wrapped_callable_names() -> None:
    def first(value: int) -> int:
        return value

    def second(value: int) -> int:
        return value

    first_tool = function_tool(functools.partial(first))
    second_tool = function_tool(functools.partial(second))

    assert first_tool.name == "first"
    assert second_tool.name == "second"


@pytest.mark.asyncio
async def test_partial_async_callable_object_supports_timeout() -> None:
    class AsyncCallable:
        async def __call__(self, value: int) -> int:
            return value * 2

    tool = function_tool(functools.partial(AsyncCallable()), timeout=1)

    assert tool.timeout_seconds == 1
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"value": 4}') == 8


@pytest.mark.asyncio
async def test_callable_object_resolves_class_scoped_call_annotations() -> None:
    class BaseHandler:
        class Payload(BaseModel):
            value: int

        async def __call__(self, payload: Payload) -> int:
            return payload.value

    class Handler(BaseHandler):
        pass

    tool = function_tool(Handler())

    assert tool.params_json_schema["properties"]["payload"] == {"$ref": "#/$defs/Payload"}
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"payload": {"value": 4}}') == 4


@pytest.mark.asyncio
async def test_callable_object_ignores_class_state_annotations() -> None:
    class Handler:
        value: str

        async def __call__(self, value: int) -> int:
            return value * 2

    tool = function_tool(Handler())

    assert tool.params_json_schema["properties"]["value"]["type"] == "integer"
    assert await tool.on_invoke_tool(ctx_wrapper(), '{"value": 4}') == 8


def test_function_tool_timeout_arguments_are_keyword_only() -> None:
    signature = inspect.signature(function_tool)

    assert signature.parameters["timeout"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["timeout_behavior"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["timeout_error_function"].kind is inspect.Parameter.KEYWORD_ONLY
