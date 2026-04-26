"""
Compatibility wrapper for legacy scripts (train.py, benchmark.py, inference.py).

The canonical OpenEnv-compliant implementation lives in server/environment.py.
This module keeps old root imports working while exposing an Environment subclass
to validators that inspect environment.py directly.
"""

from typing import Any, Optional

from server.environment import IncidentResponseEnvironment as _BaseEnvironment


class IncidentResponseEnvironment(_BaseEnvironment):
    """
    Compatibility shim for scripts using the legacy root environment API.

    Inherits from the OpenEnv-compliant implementation in server.environment and
    accepts both call styles:
    - Legacy: step(episode_id, action_dict)
    - Legacy keyword: step(episode_id=..., action_dict=...)
    - OpenEnv: step(action, timeout_s=None, episode_id=...)
    """

    def step(
        self,
        episode_id_or_action: Any = None,
        action_dict: Optional[dict] = None,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> dict:
        """Execute an action using either the legacy or OpenEnv step signature."""
        episode_id = kwargs.pop("episode_id", None)

        if action_dict is not None:
            if episode_id is None:
                if not isinstance(episode_id_or_action, str):
                    raise TypeError(
                        "Legacy step() requires step(episode_id, action_dict) "
                        "or step(episode_id=..., action_dict=...)."
                    )
                episode_id = episode_id_or_action
            return super().step(action_dict, timeout_s=timeout_s, episode_id=episode_id, **kwargs)

        if isinstance(episode_id_or_action, str):
            raise TypeError(
                "Missing action_dict for legacy step(episode_id, action_dict) call."
            )

        return super().step(
            episode_id_or_action,
            timeout_s=timeout_s,
            episode_id=episode_id,
            **kwargs,
        )
