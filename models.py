from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class EmailItem(BaseModel):
    email_id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    sla_deadline_minutes: Optional[int] = None

class EmailTriageAction(BaseModel):
    action_type: str
    category: Optional[str] = None
    priority: Optional[int] = None
    reply_text: Optional[str] = None
    email_id: Optional[str] = None

class EmailTriageObservation(BaseModel):
    current_email: Optional[EmailItem] = None
    inbox: Optional[List[EmailItem]] = None
    step_count: int = 0
    max_steps: int = 1
    episode_id: str = ""
    task_name: str = ""
    instructions: str = ""
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}

@dataclass
class EpisodeState:
    episode_id: str = ""
    task_name: str = ""
    step_count: int = 0
    max_steps: int = 1
    done: bool = False
    cumulative_reward: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
