"""A failed rank's own reason next to its outputs (<run>/rank-XX/failure.json), with credentials masked.

Nothing here may raise into the caller's error path."""
import json
import os
from pathlib import Path
import re
import time

SECRET_NAME = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.I)
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),                                   # Hugging Face tokens
    re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])"),                  # W&B API keys
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)((?:token|api[_-]?key|secret|password|authorization)[\"']?\s*[=:]\s*[\"']?)[^\s\"',}]{6,}"),
]
MASK = "[REDACTED]"


def redact(text, environ=None):
    """Mask the values of credential-like environment variables and token-shaped strings."""
    environ = os.environ if environ is None else environ
    text = str(text)
    for name, value in environ.items():
        if SECRET_NAME.search(name) and value and len(value) >= 6:
            text = text.replace(value, MASK)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + MASK, text)
    return text


def write_rank_failure(directory, exc, traceback_text, **extra):
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "failure.json").write_text(redact(json.dumps(
            {"error": type(exc).__name__, "message": str(exc), "traceback": traceback_text, "time": time.time(), **extra},
            indent=2, default=str)) + "\n")
    except Exception:
        pass
