from __future__ import annotations

import sys

from wam_policy.utils.external import bootstrap_external_repos


def test_bootstrap_external_repos_appends_paths_once(tmp_path) -> None:
    cfg = {
        "paths": {
            "repo_root": str(tmp_path),
            "lingbot_repo": "external/lingbot-va",
            "anyflow_repo": "external/AnyFlow",
        }
    }
    original_sys_path = list(sys.path)

    try:
        first = bootstrap_external_repos(cfg, require_exists=False)
        second = bootstrap_external_repos(cfg, require_exists=False)

        assert first == second
        for path in first:
            assert sys.path.count(str(path)) == 1
    finally:
        sys.path[:] = original_sys_path
