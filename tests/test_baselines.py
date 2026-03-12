import pytest

from romanian_whist.agents.baselines import BidPlayHeuristicAgent, SafeHeuristicAgent


@pytest.mark.parametrize("agent_cls", [SafeHeuristicAgent, BidPlayHeuristicAgent])
def test_bidding_heuristics_accept_no_trump_rounds(agent_cls) -> None:
    agent = agent_cls(seed=1)

    if agent_cls is SafeHeuristicAgent:
        action = agent._choose_bid_from_cards([12, 11, 25, 24], None, 8, list(range(9)))
    else:
        action = agent._aggressive_bid_from_cards([12, 11, 25, 24], None, 8, list(range(9)))

    assert action in range(9)
