from __future__ import annotations

from slop_code.agent_runner import AgentStateEnum
from slop_code.agent_runner import UsageTracker
from slop_code.entrypoints.problem_runner.models import ProblemState


def test_net_cost_does_not_double_count_completed_checkpoint() -> None:
    state = ProblemState(
        state=AgentStateEnum.RUNNING,
        overall_usage=UsageTracker(cost=5.970496),
        agent_usage=UsageTracker(cost=2.729951),
    )

    assert state.net_cost == 8.700447

    state.state = AgentStateEnum.COMPLETED
    state.overall_usage = UsageTracker(cost=8.700447)

    assert state.checkpoint_cost == 2.729951
    assert state.net_cost == 8.700447
