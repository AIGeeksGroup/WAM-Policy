"""Export a round's adapters as a released specialist, and optionally upload it.

``export --round rNN --output DIR`` writes the release ``specialist`` format (``wam_policy.json`` + ``adapter.pt`` =
the round's old adapters). ``publish`` additionally uploads the directory (and the run's round table) to a Hugging Face
model repository when one is given (``--repo`` or ``WAM_UPLOAD_REPO``); without a repository nothing is uploaded.
"""
from __future__ import annotations

from pathlib import Path

from wam_policy.task_rl import adapters as A
from wam_policy.task_rl.core import run as R


def export(run, round_name, output):
    adapter = run.root / 'rounds' / round_name / 'old-lora.pt'
    if not adapter.exists():
        raise SystemExit(f'{adapter} does not exist (round not finished?)')
    out = A.write_specialist(output, task=run.task, adapter_path=adapter, round_name=round_name,
                             rank=run.config['lora_rank'], alpha=run.config['lora_alpha'])
    print('specialist written:', out)
    return out


def publish(run, round_name, output, repo=None, path_in_repo=None):
    from wam_policy.common import site
    out = export(run, round_name, output)
    table = run.root / 'results/round-table.md'
    if table.exists():
        (Path(out) / 'round-table.md').write_text(table.read_text())
    repo = repo or site.get('WAM_UPLOAD_REPO')
    if not repo:
        print('no upload repository given (--repo / WAM_UPLOAD_REPO): kept locally')
        return out
    from huggingface_hub import HfApi
    target = path_in_repo or f'stage2_1_specialists/{run.task}'
    HfApi().upload_folder(repo_id=repo, folder_path=str(out), path_in_repo=target, repo_type='model',
                          commit_message=f'Stage 2.1 specialist {run.task} {round_name}')
    print(f'uploaded to {repo}:{target}')
    return out
