import logging
from deal_agent_framework import DealAgentFramework

logging.getLogger().setLevel(logging.INFO)

_agent_framework = None

def get_agent_framework():
    global _agent_framework

    if _agent_framework is None:
        DealAgentFramework.reset_memory()
        _agent_framework = DealAgentFramework()
        _agent_framework.init_agents_as_needed()

    return _agent_framework
