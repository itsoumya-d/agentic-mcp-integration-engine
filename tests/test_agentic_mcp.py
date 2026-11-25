import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_mcp.api import router, get_registry, get_orchestrator
from agentic_mcp.models import ExecutionStatus, ToolCall, WebhookPayload
from agentic_mcp.orchestrator import AgenticOrchestrator
from agentic_mcp.registry import MCPToolRegistry


@pytest.fixture
def clean_registry():
    reg = MCPToolRegistry()

    @reg.tool(name="get_system_status", description="Fetches system health indicators")
    def get_system_status():
        return {"status": "HEALTHY", "active_nodes": 4, "latency_ms": 12}

    @reg.tool(name="send_notification", description="Dispatches notification alert")
    async def send_notification(message: str, level: str = "info"):
        return {"delivered": True, "target": "monitoring-bus", "level": level}

    return reg


@pytest.mark.asyncio
async def test_mcp_registry_registration_and_execution(clean_registry):
    tools = clean_registry.list_tools()
    assert len(tools) == 2
    tool_names = [t.name for t in tools]
    assert "get_system_status" in tool_names
    assert "send_notification" in tool_names

    # Test sync tool execution
    call_sync = ToolCall(call_id="call-1", tool_name="get_system_status", arguments={})
    res_sync = await clean_registry.execute_tool(call_sync)
    assert res_sync.success is True
    assert res_sync.output["status"] == "HEALTHY"

    # Test async tool execution
    call_async = ToolCall(call_id="call-2", tool_name="send_notification", arguments={"message": "alert_ok", "level": "warn"})
    res_async = await clean_registry.execute_tool(call_async)
    assert res_async.success is True
    assert res_async.output["delivered"] is True


@pytest.mark.asyncio
async def test_mcp_registry_unregistered_tool():
    reg = MCPToolRegistry()
    call = ToolCall(call_id="call-fail", tool_name="unknown_tool", arguments={})
    res = await reg.execute_tool(call)
    assert res.success is False
    assert "not registered" in res.error


@pytest.mark.asyncio
async def test_orchestrator_multi_step_resolution(clean_registry):
    orchestrator = AgenticOrchestrator(registry=clean_registry, max_steps=4)
    response = await orchestrator.run("fetch system status immediately")

    assert response.status == ExecutionStatus.COMPLETED
    assert response.total_steps == 2
    assert "Workflow completed successfully" in response.final_output
    assert len(response.steps) == 2
    assert response.steps[0].tool_calls[0].tool_name == "get_system_status"
    assert response.steps[0].tool_results[0].success is True


@pytest.mark.asyncio
async def test_orchestrator_max_steps_containment():
    reg = MCPToolRegistry()

    # Planner that keeps issuing calls without concluding
    def infinite_planner(prompt, steps):
        return {
            "thought": "Keep cycling",
            "tool_calls": [ToolCall(call_id=f"c-{len(steps)}", tool_name="noop", arguments={})],
            "final_output": None,
        }

    orchestrator = AgenticOrchestrator(registry=reg, planner_fn=infinite_planner, max_steps=3)
    response = await orchestrator.run("infinite test")

    assert response.status == ExecutionStatus.MAX_STEPS_EXCEEDED
    assert response.total_steps == 3
    assert "exceeded maximum allowed steps" in response.error


def test_fastapi_webhook_and_run_endpoints():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    reg = get_registry()
    @reg.tool(name="get_system_status", description="Status tool")
    def get_status():
        return {"status": "ALL_GREEN"}

    payload = {
        "event_type": "status_check",
        "source": "monitoring_service",
        "data": {"metric": "uptime"},
        "correlation_id": "corr-uuid-101"
    }

    # Trigger webhook
    resp = client.post("/v1/webhooks/trigger", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "COMPLETED"

    # Query status by correlation_id
    resp_status = client.get("/v1/runs/corr-uuid-101")
    assert resp_status.status_code == 200
    run_data = resp_status.json()
    assert run_data["status"] == "COMPLETED"
    assert run_data["total_steps"] >= 1
