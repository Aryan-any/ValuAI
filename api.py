from fastapi import FastAPI
from typing import List

from agents.deals import Opportunity
from service import run_pricer_cycle

app = FastAPI(title="Pricer Agent API")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/run", response_model=List[Opportunity])
def run_agents():
    """
    Trigger one full agent planning cycle.
    """
    return run_pricer_cycle()
