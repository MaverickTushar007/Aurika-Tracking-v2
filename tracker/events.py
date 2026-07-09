# tracker/events.py
"""
Aurika Tracking v2 — Structured Event Engine
Defines the business and lifecycle event classes for track behavioral processing.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class BaseEvent:
    """Canonical representation of an event in the Aurika analytics timeline."""
    frame: int
    timestamp: float
    track_id: int
    event: str = field(init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.event = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Returns flat dictionary serialization compatible with event logger exports."""
        return {
            "Frame": self.frame,
            "Timestamp": round(self.timestamp, 3),
            "Track ID": self.track_id,
            "Event": self.event,
            "Metadata": str(self.metadata)
        }

@dataclass
class TrackCreated(BaseEvent):
    """Fired when a new track is first registered."""
    pass

@dataclass
class TrackConfirmed(BaseEvent):
    """Fired when a track passes confirmation thresholds."""
    pass

@dataclass
class TrackLost(BaseEvent):
    """Fired when a track transitions to temp occluded state."""
    pass

@dataclass
class TrackRecovered(BaseEvent):
    """Fired when an occluded track is successfully re-associated."""
    pass

@dataclass
class TrackArchived(BaseEvent):
    """Fired when a track is exited and archived."""
    pass

@dataclass
class EnteredRestaurant(BaseEvent):
    """Fired when a target crosses the restaurant entrance corridor."""
    pass

@dataclass
class ExitedRestaurant(BaseEvent):
    """Fired when a target leaves the restaurant boundary."""
    pass

@dataclass
class ZoneEntered(BaseEvent):
    """Fired when a target moves into a polygon zone."""
    pass

@dataclass
class ZoneExited(BaseEvent):
    """Fired when a target moves out of a polygon zone."""
    pass

@dataclass
class RoleAssigned(BaseEvent):
    """Fired when a track is first classified (e.g. Staff or Customer)."""
    pass

@dataclass
class RoleChanged(BaseEvent):
    """Fired when a track role changes dynamically."""
    pass

@dataclass
class WaitingStarted(BaseEvent):
    """Fired when waiting timeline begins (e.g. enters waiting area)."""
    pass

@dataclass
class WaitingFinished(BaseEvent):
    """Fired when waiting timeline completes."""
    pass

@dataclass
class DiningStarted(BaseEvent):
    """Fired when dining timeline starts."""
    pass

@dataclass
class DiningFinished(BaseEvent):
    """Fired when dining timeline completes."""
    pass
