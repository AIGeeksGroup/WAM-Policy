from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from wam_policy.opd.buffers import OnPolicyChunkRecord
from wam_policy.utils.paths import resolve_repo_root


class RoboTwinAdapter:
    """Adapter boundary for server-side RoboTwin rollout integration."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.repo_root = resolve_repo_root(cfg)

    def make_env(self) -> Any:
        raise NotImplementedError(
            "RoboTwin rollout must run on the server/compute node with external/RoboTwin installed."
        )

    def rollout_student_episode(self, *args: Any, **kwargs: Any) -> list[OnPolicyChunkRecord]:
        del args, kwargs
        raise NotImplementedError(
            "RoboTwin rollout must run on the server/compute node with external/RoboTwin installed."
        )

    def save_rollout_manifest(
        self,
        records: Iterable[OnPolicyChunkRecord],
        path: str | Path,
    ) -> Path:
        output_path = Path(path).expanduser()
        if not output_path.is_absolute():
            output_path = self.repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), sort_keys=True))
                handle.write("\n")
        return output_path.resolve()
