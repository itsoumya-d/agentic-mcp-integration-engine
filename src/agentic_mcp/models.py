from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MAX_STEPS_EXCEEDED = "MAX_STEPS_EXCEEDED"


class ToolDefinition(BaseModel):
    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Actionable capability description")
    parameters_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema for inputs")


class ToolCall(BaseModel):
    call_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    output: Any = None
    error: Optional[str] = None
    success: bool = True


class TrajectoryStep(BaseModel):
    step_number: int
    thought: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)


class AgentResponse(BaseModel):
    final_output: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    steps: List[TrajectoryStep] = Field(default_factory=list)
    total_steps: int = 0
    error: Optional[str] = None


class WebhookPayload(BaseModel):
    event_type: str
    source: str
    data: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: str
