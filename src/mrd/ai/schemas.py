from typing import Literal
from pydantic import BaseModel

class AIInvestigation(BaseModel):

    severity:Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]
    root_cause:str
    evidence: list[str]
    affected_areas: list[str]
    recommendations:list[str]
    confidence: float


