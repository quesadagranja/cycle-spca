"""Calendar-Structured Graph-Fused Sparse PCA."""

from .graph import (
    DEFAULT_CALENDAR_SHAPE,
    ToroidalCalendarGraph,
    flatten_calendar,
    reshape_calendar,
)
from .model import CalendarGraphFusedSparsePCA
from .stability import ComponentMatch, fit_restarts, match_components

__all__ = [
    "CalendarGraphFusedSparsePCA",
    "ComponentMatch",
    "DEFAULT_CALENDAR_SHAPE",
    "ToroidalCalendarGraph",
    "fit_restarts",
    "flatten_calendar",
    "match_components",
    "reshape_calendar",
]

__version__ = "0.1.0"
