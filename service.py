from typing import List
from agents.deals import Opportunity
from core import get_agent_framework


def run_pricer_cycle() -> List[Opportunity]:
    """
    Programmatic entrypoint for running one full agent cycle.
    This is what APIs, schedulers, and automation will call.
    """

    framework = get_agent_framework()
    opportunities = framework.run()
    return opportunities


if __name__ == "__main__":
    results = run_pricer_cycle()
    for opp in results:
        print(opp)
