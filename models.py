from pydantic import BaseModel
from typing import Optional

class ModerationAction(BaseModel):
    action: str  # ALLOW, LABEL_WARNING, REDUCE_REACH, DELETE, ESCALATE

class ModerationObservation(BaseModel):
    misinfo_probability: float
    virality_score: float
    spread_velocity: float
    report_count: int
    trusted_report_count: int
    reporter_trust: float
    user_credibility: float
    is_repeat_offender: bool
    factcheck_confidence: float
    environmental_warning: str  
    content_category: str
    reward: Optional[float] = 0.0
    done: Optional[bool] = False