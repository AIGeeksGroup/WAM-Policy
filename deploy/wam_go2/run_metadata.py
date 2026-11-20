"""Startup-only provenance for offline reproduction of a POLICY trace."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_metadata(checkpoint, parameters):
    """Hash actual loaded code/checkpoint, including uncommitted source edits."""
    package = Path(__file__).resolve().parent
    files = sorted(package.rglob('*.py'))
    # The effective parameter values below remain authoritative after install.
    for directory in ('config', 'launch', 'scripts'):
        root = package.parent / directory
        if root.exists():
            files.extend(p for p in sorted(root.rglob('*'))
                         if p.is_file() and p.suffix in ('.py', '.sh', '.yaml', '.xml'))
    versions = {}
    for name in ('numpy', 'scipy', 'osqp', 'torch', 'onnxruntime'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return json.dumps(dict(
        schema_version=1, python=platform.python_version(), dependencies=versions,
        checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256_file(checkpoint),
        parameters=parameters,
        source_sha256={str(p.relative_to(package.parent)): sha256_file(p) for p in files}),
        sort_keys=True)
