"""
Global patch registry — shared across all explorer instances within a pipeline run.

Prevents duplicate ROI captioning across parallel explorers and across supervisor
iterations. Also exposes a human-readable summary that is injected into both the
explorer and supervisor prompts so the agents know what has already been viewed.
"""
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class _PatchKey:
    x: int
    y: int
    magnification: float


class GlobalPatchRegistry:
    """Thread-safe registry of visited patches, shared across all explorer instances."""

    def __init__(self):
        self._visited: set[_PatchKey] = set()
        self._descriptions: dict[_PatchKey, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #
    def reset(self):
        """Call once at the start of each pipeline run (per slide)."""
        with self._lock:
            self._visited.clear()
            self._descriptions.clear()

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #
    def register(self, x: int, y: int, magnification: float, description: str = ""):
        """Mark a patch as visited. Idempotent."""
        key = _PatchKey(x=int(x), y=int(y), magnification=float(magnification))
        with self._lock:
            self._visited.add(key)
            if description and key not in self._descriptions:
                self._descriptions[key] = description

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #
    def is_visited(self, x: int, y: int, magnification: float) -> bool:
        key = _PatchKey(x=int(x), y=int(y), magnification=float(magnification))
        return key in self._visited

    def get_visited_list(self) -> list[dict]:
        with self._lock:
            return [
                {"x": k.x, "y": k.y, "magnification": k.magnification}
                for k in sorted(self._visited, key=lambda k: (k.x, k.y, k.magnification))
            ]

    def summary(self, max_entries: int = 40) -> str:
        """
        Human-readable list for agent prompts.
        Capped at max_entries to avoid bloating the context window.
        """
        visited = self.get_visited_list()
        if not visited:
            return "None yet."
        lines = [
            f"  ({p['x']}, {p['y']}) @ {p['magnification']}x"
            for p in visited[:max_entries]
        ]
        if len(visited) > max_entries:
            lines.append(f"  ... and {len(visited) - max_entries} more (omitted)")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._visited)


# ------------------------------------------------------------------ #
#  Module-level singleton                                              #
# ------------------------------------------------------------------ #
_registry = GlobalPatchRegistry()


def get_registry() -> GlobalPatchRegistry:
    """Return the process-wide patch registry."""
    return _registry


def reset_registry():
    """Convenience wrapper — reset before each new slide."""
    _registry.reset()
