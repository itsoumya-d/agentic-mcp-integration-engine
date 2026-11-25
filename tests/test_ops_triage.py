from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_mcp.api import router
from agentic_mcp.ops_triage import (
    ApiTelemetryEvent,
    GenAIOpsTriager,
    IncidentCategory,
    K8sWorkloadSnapshot,
    Severity,
)


def test_triage_usage_auth_and_rate_limit():
    workload = K8sWorkloadSnapshot(
        namespace="genai-prod",
        deployment="llm-gateway",
        pod_name="llm-gateway-6f7b",
        phase="Running",
        ready_replicas=3,
        desired_replicas=3,
    )
    telemetry = ApiTelemetryEvent(
        trace_id="tr-auth-403",
        endpoint="/v1/completions",
        status_code=403,
        error_code="RBAC_FORBIDDEN",
        user_role="viewer",
    )
    report = GenAIOpsTriager.diagnose(workload, telemetry)
    assert report.category == IncidentCategory.USAGE_ISSUE
    assert report.severity == Severity.P3_STANDARD
    assert report.escalate_to_engineering is False
    assert "RBAC" in report.recommended_action


def test_triage_k8s_oom_operational_problem():
    workload = K8sWorkloadSnapshot(
        namespace="genai-prod",
        deployment="embedding-worker",
        pod_name="embedding-worker-9c2a",
        phase="CrashLoopBackOff",
        ready_replicas=0,
        desired_replicas=2,
        restart_count=6,
        termination_reason="OOMKilled",
        events=["Back-off restarting failed container", "OOMKilled exit code 137"],
    )
    telemetry = ApiTelemetryEvent(
        trace_id="tr-k8s-503",
        endpoint="/v1/embeddings",
        status_code=503,
    )
    report = GenAIOpsTriager.diagnose(workload, telemetry)
    assert report.category == IncidentCategory.OPERATIONAL_PROBLEM
    assert report.severity == Severity.P1_CRITICAL
    assert report.escalate_to_engineering is True
    assert report.evidence["termination_reason"] == "OOMKilled"


def test_triage_k8s_pending_scheduling_failure():
    workload = K8sWorkloadSnapshot(
        namespace="genai-prod",
        deployment="reranker-svc",
        pod_name="reranker-svc-0",
        phase="Pending",
        ready_replicas=1,
        desired_replicas=2,
        events=["0/6 nodes are available: 6 Insufficient memory (FailedScheduling)."],
    )
    telemetry = ApiTelemetryEvent(trace_id="tr-sched", endpoint="/v1/rerank", status_code=200)
    report = GenAIOpsTriager.diagnose(workload, telemetry)
    assert report.category == IncidentCategory.OPERATIONAL_PROBLEM
    assert report.severity == Severity.P2_HIGH


def test_triage_product_bug_escalation_with_traceback():
    workload = K8sWorkloadSnapshot(
        namespace="genai-prod",
        deployment="prompt-router",
        pod_name="prompt-router-4a1d",
        phase="Running",
        ready_replicas=2,
        desired_replicas=2,
        log_lines=[
            "INFO request trace=tr-bug-500 started",
            "ERROR Traceback (most recent call last): KeyError: 'tenant_config'",
        ],
    )
    telemetry = ApiTelemetryEvent(
        trace_id="tr-bug-500",
        endpoint="/v1/route",
        status_code=500,
    )
    report = GenAIOpsTriager.diagnose(workload, telemetry)
    assert report.category == IncidentCategory.PRODUCT_BUG
    assert report.escalate_to_engineering is True
    assert len(report.evidence["log_excerpts"]) == 1


def test_fastapi_ops_triage_endpoint():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.post(
        "/v1/ops/triage",
        json={
            "workload": {
                "namespace": "genai-prod",
                "deployment": "llm-gateway",
                "pod_name": "llm-gateway-1",
                "phase": "Running",
                "ready_replicas": 2,
                "desired_replicas": 2,
            },
            "telemetry": {
                "trace_id": "tr-api-429",
                "endpoint": "/v1/chat",
                "status_code": 429,
                "error_code": "QUOTA_EXCEEDED",
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "USAGE_ISSUE"
    assert body["escalate_to_engineering"] is False
