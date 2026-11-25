import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional
from agentic_mcp.models import ToolCall, ToolDefinition, ToolResult


class MCPToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[..., Any]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameters_schema: Optional[Dict[str, Any]] = None,
        handler: Optional[Callable[..., Any]] = None,
    ) -> None:
        defn = ToolDefinition(
            name=name,
            description=description,
            parameters_schema=parameters_schema or {"type": "object", "properties": {}},
        )
        self._tools[name] = defn
        if handler:
            self._handlers[name] = handler

    def tool(self, name: str, description: str, parameters_schema: Optional[Dict[str, Any]] = None):
        """Decorator for registering functions as MCP tools."""
        def decorator(fn: Callable[..., Any]):
            self.register_tool(name=name, description=description, parameters_schema=parameters_schema, handler=fn)
            return fn
        return decorator

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    async def execute_tool(self, call: ToolCall) -> ToolResult:
        if call.tool_name not in self._handlers:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=f"Tool '{call.tool_name}' is not registered in MCP registry",
                success=False,
            )

        handler = self._handlers[call.tool_name]
        try:
            if inspect.iscoroutinefunction(handler):
                res = await asyncio.wait_for(handler(**call.arguments), timeout=15.0)
            else:
                res = handler(**call.arguments)

            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=res,
                error=None,
                success=True,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=f"Execution timeout (15s) exceeded for tool '{call.tool_name}'",
                success=False,
            )
        except Exception as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=None,
                error=f"Execution failed: {type(exc).__name__}: {str(exc)}",
                success=False,
            )
