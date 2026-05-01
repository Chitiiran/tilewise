"""OpenSpiel MCTS study driven by the catan_bot Rust engine."""

__version__ = "0.2.0"  # bumped for v2 engine integration

# Read action space size dynamically from the engine — was hardcoded as 206
# in v1, but v2's full Catan rules expand to 280 (added trades + dev cards).
from catan_bot import _engine as _e
ACTION_SPACE_SIZE = _e.action_space_size()
del _e
