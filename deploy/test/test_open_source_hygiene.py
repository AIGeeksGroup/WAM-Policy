"""deploy/ is published as open source: every text file stays free of non-English text, development
notes and machine-specific setup. Patterns are assembled at run time so this file does not match itself."""
import re
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", ".pytest_cache", "result", "data", "build", "install", "log", "third_party"}
TEXT_SUFFIXES = {".py", ".sh", ".bash", ".md", ".txt", ".yaml", ".yml", ".json", ".xml", ".cfg", ".ini",
                 ".conf", ".service", ".env", ".example", ""}


def text_files():
    for path in sorted(DEPLOY.rglob("*")):
        if not path.is_file() or SKIP_DIRS & set(path.relative_to(DEPLOY).parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name.endswith(".example.env"):
            try:
                yield path, path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue


FILES = list(text_files())
CJK = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in ((0x3000, 0x303F), (0x4E00, 0x9FFF), (0xFF00, 0xFFEF))) + "]")


def j(*parts):
    return "".join(parts)


# development history, internal names and machine-specific setup
FORBIDDEN = [
    (j(r"\bown", r"er\b"), "quotes / decisions of a person"),
    (j("HAND", "OFF"), "internal hand-over notes"),
    (j("AGENTS", r"\.md"), "internal working rules"),
    (j("this ", "laptop"), "one machine's setup"),
    (j("enx6c6e", "070dac93"), "one machine's network interface"),
    (j("id_ed25519", "_go2"), "one machine's SSH key"),
    (j("src", "_ext"), "internal checkout layout"),
    (j("runs/", "audit"), "internal experiment records"),
    (j("SFT", "5000"), "internal checkpoint name"),
    (j("online", "_service"), "internal module"),
    (j("review ", "finding"), "review notes"),
    (j("Go2 ", "Air"), "robot variant"),
    (j(r"go2_cam(?!era)", ""), "robot-side camera capture"),
    (j("go2_", "service"), "robot service switching"),
    (j("video", "_hub"), "robot service switching"),
    (j("para", "miko"), "SSH to the robot"),
    (j(r"\bL[12]", r"\+[A-Za-z]"), "wireless-remote button combos"),
    (j("third_", "view"), "third-person recording"),
]
# IPv4 addresses other than loopback, "any" and the reserved documentation ranges (RFC 5737)
IPV4 = re.compile(r"(?<![0-9.])(?!127\.0\.0\.1\b|0\.0\.0\.0\b|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)"
                  r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])")
# the same as 8 hex digits, e.g. a chrony reference ID: 10/8, 172.16/12, 192.168/16
HEX_IPV4 = re.compile(r"(?<![0-9A-Za-z])(?:0A[0-9A-F]{6}|AC1[0-9A-F]{5}|C0A8[0-9A-F]{4})(?![0-9A-Za-z])")
DATE = re.compile(j(r"(?<![0-9])20[0-9]{2}-[01][0-9]-[0-3][0-9](?![0-9])"))
DATE_IS_DATA = {"test/test_pair_report.py", "test/test_go2_camera_guard.py", "test/fixtures/static_reference_session.json"}


def test_there_are_files_to_check():
    names = {p.relative_to(DEPLOY).as_posix() for p, _ in FILES}
    assert {"README.md", "run_wam.sh", "wam_go2/nodes/controller_node.py"} <= names


@pytest.mark.parametrize("path,text", FILES, ids=[p.relative_to(DEPLOY).as_posix() for p, _ in FILES])
def test_english_only_and_no_development_notes(path, text):
    rel = path.relative_to(DEPLOY).as_posix()
    hits = [f"line {text.count(chr(10), 0, m.start()) + 1}: non-English text" for m in CJK.finditer(text)][:3]
    if rel != "test/test_open_source_hygiene.py":
        for pattern, why in FORBIDDEN:
            for m in re.finditer(pattern, text):
                hits.append(f"line {text.count(chr(10), 0, m.start()) + 1}: {m.group(0)!r} ({why})")
        for m in IPV4.finditer(text):
            hits.append(f"line {text.count(chr(10), 0, m.start()) + 1}: address {m.group(0)} (robot or host address)")
        for m in HEX_IPV4.finditer(text):
            hits.append(f"line {text.count(chr(10), 0, m.start()) + 1}: {m.group(0)} (hex-encoded private address)")
        if rel not in DATE_IS_DATA:
            for m in DATE.finditer(text):
                hits.append(f"line {text.count(chr(10), 0, m.start()) + 1}: date {m.group(0)} (development history)")
    assert not hits, f"{rel}: " + "; ".join(hits[:10])


def test_the_local_configuration_and_weights_stay_out_of_git():
    rules = (DEPLOY / ".gitignore").read_text().splitlines()
    for rule in ("config/deploy.env", "policies/quadruped/*.pt", "/result/", "third_party/"):
        assert rule in rules, rule
    assert (DEPLOY / "config" / "deploy.example.env").is_file()
