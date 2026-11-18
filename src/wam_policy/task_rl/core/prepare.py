"""Prepare the training scenes of one task (CPU): unpack the QLM-Bench Task RL inputs into a data root and write a
cases directory the run reads.

Inputs (``<QLM-Bench>/inputs/task_rl/<task>/``): ``workspace-files.tar.gz`` (scene packs without their large meshes, case
files, small resources; paths relative to a data root, written as ``@WORKSPACE@/...`` in the index files),
``scene-files.json`` (every file a run reads below the data root, with sha256: the scene meshes and the RAMBO
controller runtime files ``assets/rambo-go2-controller/iteration-2000/runtime/{model_2000.pt,agent.yaml,env.yaml}``),
and for open_room_door ``cases/`` + ``cases-index.json``.

Steps: (1) unpack the archive below ``--data-root`` (existing files are kept; contents unchanged, since QLM-Bench
checks the scene resources against their checksums); (2) files of ``scene-files.json`` still missing (the large meshes and the controller,
which the archive does not contain) are linked from ``--reuse <dir>`` (a tree with the same layout) or downloaded with ``--download`` from the QLM-Bench
dataset repository (``QLM_DATASET_REPO`` / ``QLM_DATASET_REVISION`` of the site configuration, or ``--repo`` /
``--revision``), matched by sha256; (3) ``--out``: ``cases-index.json`` (training cases with absolute scene and case
paths and the controller, ``@WORKSPACE@`` resolved to the data root) and a copy of the case files. ``--controller`` (or ``RAMBO_CONTROLLER``) replaces the
controller named by the inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

from wam_policy.task_rl.core import run as R

PLACEHOLDER = '@WORKSPACE@'
CASES_SCHEMA = 'wam-policy-task-rl-cases-v1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def unpack(archive, root):
    """Extract (keeping existing files); returns the member names (regular files)."""
    root = Path(root).resolve()
    names = []
    with tarfile.open(archive, 'r:gz') as tar:
        for m in tar.getmembers():
            target = (root / m.name).resolve()
            if not str(target).startswith(str(root) + os.sep) or m.issym():
                raise ValueError(f'Unsafe archive member {m.name}')
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not (m.isfile() or m.islnk()):
                continue
            names.append(m.name)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if m.islnk():                      # identical files are stored once (hard links inside the archive)
                source = (root / m.linkname).resolve()
                if not str(source).startswith(str(root) + os.sep) or not source.is_file():
                    raise ValueError(f'Unsafe archive link {m.name} -> {m.linkname}')
                shutil.copyfile(source, target)
                continue
            with tar.extractfile(m) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
    # Files keep their published bytes: QLM-Bench checks scene-pack resources against their checksums, and
    # ``@WORKSPACE@`` inside them is provenance only. Paths this module writes are resolved by ``resolve``.
    return names


def resolve(path, root):
    """A stored path: ``@WORKSPACE@`` -> the data root; an absolute path that does not exist is looked up below the data
    root by its longest existing suffix; a relative path is relative to the data root."""
    root = Path(root).resolve()
    s = str(path).replace(PLACEHOLDER, str(root))
    p = Path(s)
    if not p.is_absolute():
        return root / p
    if p.exists():
        return p
    parts = p.parts
    for i in range(1, len(parts)):
        cand = root.joinpath(*parts[i:])
        if cand.exists():
            return cand
    return p


def link_missing(files, root, reuse):
    """Hard-link (or copy) files missing below ``root`` from ``reuse`` (same relative layout); QLM-Bench refuses scene
    resources that are symlinks out of their pack. Returns (linked, still missing)."""
    linked, missing = 0, []
    for rel in files:
        p = root / rel
        if p.exists():
            continue
        src = Path(reuse) / rel
        if src.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.is_symlink():
                p.unlink()
            try:
                os.link(src.resolve(), p)
            except OSError:
                shutil.copyfile(src.resolve(), p)
            linked += 1
        else:
            missing.append(rel)
    return linked, missing


def download_missing(files, root, repo, revision, store):
    """Fetch missing files from the dataset repository by sha256 (LFS files under ``assets/``)."""
    from huggingface_hub import HfApi, hf_hub_download
    todo = {rel: want for rel, want in files.items() if not (root / rel).exists()}
    if not todo:
        return 0, []
    api, by_sha = HfApi(), {}
    for x in api.list_repo_tree(repo, path_in_repo='assets', repo_type='dataset', revision=revision, recursive=True):
        if not hasattr(x, 'size'):
            continue
        if getattr(x, 'lfs', None):
            by_sha.setdefault(x.lfs.sha256, x.path)
        elif '/' in x.path:          # small (non-LFS) files: hash after download
            f = hf_hub_download(repo, x.path, revision=revision, repo_type='dataset', local_dir=str(store))
            by_sha.setdefault(sha256(f), x.path)
    got, missing = 0, []
    for rel, want in sorted(todo.items()):
        if want not in by_sha:
            missing.append(rel)
            continue
        f = Path(hf_hub_download(repo, by_sha[want], revision=revision, repo_type='dataset', local_dir=str(store)))
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(f, p)
        except OSError:
            shutil.copyfile(f, p)
        got += 1
    return got, missing


def main(args):
    from wam_policy.common import site
    from wam_policy.task_rl.tasks import get
    get(args.task)
    inputs, root, out = Path(args.inputs).resolve(), Path(args.data_root).resolve(), Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    names = unpack(inputs / 'workspace-files.tar.gz', root)
    print(f'unpacked {len(names)} files into {root}', flush=True)
    scene = R.load_json(inputs / 'scene-files.json')['files']
    scene = {rel.replace(PLACEHOLDER + '/', ''): v for rel, v in scene.items()}
    missing = [rel for rel in scene if not (root / rel).exists()]
    if missing and args.reuse:
        linked, missing = link_missing(missing, root, args.reuse)
        print(f'linked {linked} files from {args.reuse}', flush=True)
    if missing and args.download:
        repo = args.repo or site.get('QLM_DATASET_REPO')
        revision = args.revision or site.get('QLM_DATASET_REVISION')
        got, missing = download_missing({k: scene[k] for k in missing}, root, repo, revision, root / '.download')
        print(f'downloaded {got} files from {repo}@{revision}', flush=True)
    if missing:
        print(f'WARNING: {len(missing)} scene files missing below {root}, e.g. {missing[:3]}', flush=True)
    if (inputs / 'cases-index.json').exists():
        index_path = inputs / 'cases-index.json'
    else:
        found = [n for n in names if n.endswith('/cases-index.json')]
        if len(found) != 1:
            raise SystemExit(f'expected one cases-index.json in the archive, found {found}')
        index_path = root / found[0]
    ix = R.load_json(index_path)
    (out / 'cases').mkdir(parents=True, exist_ok=True)
    cases = []
    for c in ix['cases']:
        if c.get('task_id', args.task) != args.task or not c.get('model_path'):
            continue
        src = index_path.parent / 'cases' / f"{c['case_id']}.json"
        if not src.exists():
            continue                                   # held-out cases are listed but never written
        dst = out / 'cases' / src.name
        shutil.copyfile(src, dst)
        cases.append(dict(case_id=c['case_id'], split=c['split'], case=str(dst), model=str(resolve(c['model_path'], root)),
                          horizon_s=c['horizon_s']))
    override = getattr(args, 'controller', None) or os.environ.get('RAMBO_CONTROLLER')
    controller = Path(override).expanduser().resolve() if override else resolve(ix['controller_checkpoint'], root)
    if not controller.is_file():
        raise SystemExit(f'RAMBO controller not found: {controller}. Fetch the scene files (--download or --reuse) '
                         'or pass --controller / set RAMBO_CONTROLLER to model_2000.pt')
    R.write_json(out / 'cases-index.json', {'schema': CASES_SCHEMA, 'task': args.task, 'data_root': str(root),
                                            'controller': str(controller), 'cases': cases})
    print(f'{len(cases)} cases -> {out / "cases-index.json"} ({time.time() - t0:.0f} s)', flush=True)
