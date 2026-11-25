from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IncidentCategory(str, Enum):
    USAGE_ISSUE = "USAGE_ISSUE"
    OPERATIONAL_PROBLEM = "OPERATIONAL_PROBLEM"
    PRODUCT_BUG = "PRODUCT_BUG"


class Severity(str, Enum):
    P1_CRITICAL = "P1_CRITICAL"
    P2_HIGH = "P2_HIGH"
    P3_STANDARD = "P3_STANDARD"


class K8sWorkloadSnapshot(BaseModel):
    namespace: str
    deployment: str
    pod_name: str
    phase: str = Field(..., description="Running, Pending, Failed, CrashLoopBackOff, etc.")
    ready_replicas: int = 1
    desired_replicas: int = 1
    restart_count: int = 0
    termination_reason: Optional[str] = None
    events: List[str] = Field(default_factory=list)
    log_lines: List[str] = Field(default_factory=list)


class ApiTelemetryEvent(BaseModel):
    trace_id: str
    endpoint: str
    status_code: int
    latency_ms: float = 0.0
    error_code: Optional[str] = None
    user_role: Optional[str] = None


class TriageRequest(BaseModel):
    workload: K8sWorkloadSnapshot
    telemetry: ApiTelemetryEvent


class TriageReport(BaseModel):
    category: IncidentCategory
    severity: Severity
    escalate_to_engineering: bool
    summary: str
    recommended_action: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


class GenAIOpsTriager:
    """Correlates Kubernetes workload health, logs, events, and API telemetry."""

    _INFRA_REASONS = {"OOMKilled", "CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull"}
    _SCHEDULING_MARKERS = ("insufficient memory", "insufficient cpu", "failedscheduling", "unhealthy")
    _EXCEPTION_MARKERS = ("traceback", "keyerror", "attributeerror", "typeerror", "unhandled exception", "nullpointer")

    @classmethod
    def diagnose(cls, workload: K8sWorkloadSnapshot, telemetry: ApiTelemetryEvent) -> TriageReport:
        events_blob = " ".join(workload.events).lower()
        logs_blob = "\n".join(workload.log_lines).lower()
        reason = (workload.termination_reason or "").strip()

        # 1. Operational / Kubernetes workload problem
        degraded_replicas = workload.ready_replicas < workload.desired_replicas
        infra_crash = (
            workload.phase in {"CrashLoopBackOff", "Pending", "Failed"}
            or reason in cls._INFRA_REASONS
            or workload.restart_count >= 3
            or any(m in events_blob for m in cls._SCHEDULING_MARKERS)
        )
        if infra_crash or (telemetry.status_code in {502, 503, 504} and degraded_replicas):
            sev = Severity.P1_CRITICAL if workload.ready_replicas == 0 else Severity.P2_HIGH
            return TriageReport(
                category=IncidentCategory.OPERATIONAL_PROBLEM,
                severity=sev,
                escalate_to_engineering=sev == Severity.P1_CRITICAL,
                summary=f"Kubernetes workload '{workload.deployment}' unhealthy (phase={workload.phase}, restarts={workload.restart_count}).",
                recommended_action="Inspect pod events/limits (`kubectl describe pod`) and restore healthy replica capacity.",
                evidence={
                    "namespace": workload.namespace,
                    "pod": workload.pod_name,
                    "termination_reason": reason or None,
                    "events": workload.events[:5],
                    "trace_id": telemetry.trace_id,
                },
            )

        # 2. Usage / Configuration / Auth / Quota issue on healthy cluster
        if telemetry.status_code in {400, 401, 403, 422, 429}:
            action = (
                "Verify API key / RBAC role assignment for user."
                if telemetry.status_code in {401, 403}
                else "Advise user on token quota backoff or validate request schema."
            )
            return TriageReport(
                category=IncidentCategory.USAGE_ISSUE,
                severity=Severity.P3_STANDARD,
                escalate_to_engineering=False,
                summary=f"Client or access configuration error HTTP {telemetry.status_code} on {telemetry.endpoint} while pods are healthy.",
                recommended_action=action,
                evidence={
                    "status_code": telemetry.status_code,
                    "error_code": telemetry.error_code,
                    "user_role": telemetry.user_role,
                    "trace_id": telemetry.trace_id,
                },
            )

        # 3. Product bug: pods healthy, 5xx or stack trace in application logs
        stack_hits = [ln for ln in workload.log_lines if any(m in ln.lower() for m in cls._EXCEPTION_MARKERS)]
        if telemetry.status_code >= 500 or stack_hits:
            return TriageReport(
                category=IncidentCategory.PRODUCT_BUG,
                severity=Severity.P2_HIGH,
                escalate_to_engineering=True,
                summary=f"Application fault HTTP {telemetry.status_code} on {telemetry.endpoint} with healthy Kubernetes replicas.",
                recommended_action="Escalate reproduction bundle with trace_id and stack excerpt to GenAI Platform engineering.",
                evidence={
                    "trace_id": telemetry.trace_id,
                    "endpoint": telemetry.endpoint,
                    "status_code": telemetry.status_code,
                    "log_excerpts": stack_hits[:5] or workload.log_lines[-3:],
                },
            )

        return TriageReport(
            category=IncidentCategory.USAGE_ISSUE,
            severity=Severity.P3_STANDARD,
            escalate_to_engineering=False,
            summary=f"No fault detected on {workload.deployment} (HTTP {telemetry.status_code}, {telemetry.latency_ms:.1f}ms).",
            recommended_action="Close ticket with latency and health telemetry confirmation.",
            evidence={"trace_id": telemetry.trace_id, "latency_ms": telemetry.latency_ms},
        )
