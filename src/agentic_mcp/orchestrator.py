import asyncio
import uuid
from typing import Any, Callable, Dict, List, Optional
from agentic_mcp.models import (
    AgentResponse,
    ExecutionStatus,
    ToolCall,
    ToolResult,
    TrajectoryStep,
)
from agentic_mcp.registry import MCPToolRegistry


class AgenticOrchestrator:
    def __init__(
        self,
        registry: MCPToolRegistry,
        planner_fn: Optional[Callable[[str, List[TrajectoryStep]], Dict[str, Any]]] = None,
        max_steps: int = 5,
    ):
        self.registry = registry
        self.planner_fn = planner_fn or self._default_deterministic_planner
        self.max_steps = max_steps

    def _default_deterministic_planner(
        self, prompt: str, steps: List[TrajectoryStep]
    ) -> Dict[str, Any]:
        """Simple rule-based deterministic planner for testable workflows."""
        # If no steps run yet, inspect query keywords
        if not steps:
            p_lower = prompt.lower()
            if "status" in p_lower or "fetch" in p_lower:
                return {
                    "thought": "Query indicates a status lookup is required.",
                    "tool_calls": [
                        ToolCall(
                            call_id=str(uuid.uuid4())[:8],
                            tool_name="get_system_status",
                            arguments={},
                        )
                    ],
                    "final_output": None,
                }
            if "notify" in p_lower or "webhook" in p_lower:
                return {
                    "thought": "Action requires sending notification webhook.",
                    "tool_calls": [
                        ToolCall(
                            call_id=str(uuid.uuid4())[:8],
                            tool_name="send_notification",
                            arguments={"message": prompt, "level": "info"},
                        )
                    ],
                    "final_output": None,
                }

        # If previous step had successful tool execution, summarize and conclude
        last_step = steps[-1] if steps else None
        if last_step and last_step.tool_results:
            success = all(r.success for r in last_step.tool_results)
            if success:
                res_summaries = [f"{r.tool_name}: {r.output}" for r in last_step.tool_results]
                return {
                    "thought": "Tool executions completed successfully. Formulating final response.",
                    "tool_calls": [],
                    "final_output": f"Workflow completed successfully. Results: {'; '.join(res_summaries)}",
                }
            else:
                err_summaries = [f"{r.tool_name}: {r.error}" for r in last_step.tool_results if not r.success]
                return {
                    "thought": "Tool execution encountered errors. Terminating with failure report.",
                    "tool_calls": [],
                    "final_output": f"Workflow aborted due to errors: {'; '.join(err_summaries)}",
                }

        return {
            "thought": "No further tool execution required.",
            "tool_calls": [],
            "final_output": f"Processed prompt: {prompt}",
        }

    async def run(self, prompt: str) -> AgentResponse:
        trajectory: List[TrajectoryStep] = []

        for step_idx in range(1, self.max_steps + 1):
            plan = self.planner_fn(prompt, trajectory)
            thought = plan.get("thought", "")
            tool_calls = plan.get("tool_calls", [])
            final_output = plan.get("final_output")

            if not tool_calls and final_output is not None:
                trajectory.append(
                    TrajectoryStep(
                        step_number=step_idx,
                        thought=thought,
                        tool_calls=[],
                        tool_results=[],
                    )
                )
                return AgentResponse(
                    final_output=final_output,
                    status=ExecutionStatus.COMPLETED,
                    steps=trajectory,
                    total_steps=step_idx,
                )

            # Execute tool calls
            tool_results: List[ToolResult] = []
            for tc in tool_calls:
                res = await self.registry.execute_tool(tc)
                tool_results.append(res)

            step = TrajectoryStep(
                step_number=step_idx,
                thought=thought,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            trajectory.append(step)

            # If any tool failed hard, or if final_output is already reached
            if final_output is not None:
                return AgentResponse(
                    final_output=final_output,
                    status=ExecutionStatus.COMPLETED,
                    steps=trajectory,
                    total_steps=step_idx,
                )

        return AgentResponse(
            final_output=None,
            status=ExecutionStatus.MAX_STEPS_EXCEEDED,
            steps=trajectory,
            total_steps=self.max_steps,
            error=f"Agent exceeded maximum allowed steps ({self.max_steps}) without resolution.",
        )
