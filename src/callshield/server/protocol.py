"""Typed message models and protocol definitions for Call Shield server."""

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = "ok"
    detector: str


class ScorePoint(BaseModel):
    """A single evaluation point in a scoring timeline."""

    t: float
    score: float
    level: str


class ScoreResponse(BaseModel):
    """Response returned by /api/score containing a full timeline."""

    timeline: list[ScorePoint]


class AlertStateMessage(BaseModel):
    """Message sent over /ws/stream indicating window evaluation state."""

    model_config = ConfigDict(extra="ignore")

    t: float
    score: float
    level: str
    flagged: int
    reason: str


class AlertBroadcastMessage(BaseModel):
    """Alert notification published to subscribed users via AlertHub and /ws/alerts."""

    model_config = ConfigDict(extra="ignore")

    t: float
    score: float
    level: str
    flagged: int
    reason: str
    alert: bool
    message: str
