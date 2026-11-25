from typing import Dict
from fastapi import APIRouter, HTTPException
from agentic_mcp.models import AgentResponse, ExecutionStatus, WebhookPayload
from agentic_mcp.ops_triage import GenAIOpsTriager, TriageReport, TriageRequest
from agentic_mcp.orchestrator import AgenticOrchestrator
from agentic_mcp.registry import MCPToolRegistry

router = APIRouter(prefix="/v1", tags=["agentic-mcp"])

_registry = MCPToolRegistry()
_orchestrator = AgenticOrchestrator(registry=_registry)
_runs: Dict[str, AgentResponse] = {}


def get_registry() -> MCPToolRegistry:
    return _registry


def get_orchestrator() -> AgenticOrchestrator:
    return _orchestrator


@router.post("/webhooks/trigger", response_model=AgentResponse)
async def handle_webhook(payload: WebhookPayload) -> AgentResponse:
    prompt = f"Event '{payload.event_type}' received from '{payload.source}'. Data: {payload.data}"
    response = await _orchestrator.run(prompt)
    _runs[payload.correlation_id] = response
    return response


@router.get("/runs/{correlation_id}", response_model=AgentResponse)
async def get_run_status(correlation_id: str) -> AgentResponse:
    if correlation_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run '{correlation_id}' not found")
    return _runs[correlation_id]


@router.post("/ops/triage", response_model=TriageReport)
async def triage_incident(req: TriageRequest) -> TriageReport:
    """Diagnose Kubernetes workload snapshots and API telemetry into actionable triage reports."""
    return GenAIOpsTriager.diagnose(req.workload, req.telemetry)
