"""Offline tests for the clock-step guard of scripts/x5/go2_camera.sh (the RDK X5
camera host script). No X5, no ROS, no robot: the functions `wall_minus_mono`
and `clock_guard` are cut out of the script and run by bash against stubbed
clocks.

Why the guard matters: chrony on the X5 steps every correction above 0.1 s
(chrony_x5.conf `makestep 0.1 -1`) and librealsense 2.58.4's device->host time
fit rejects anything more than 100 ms off for 60 s, so every chrony step must
restart the camera node, while a stalled read must not.

Two levels of stub:
- value level: `wall_minus_mono` itself is replaced by a table of per-poll
  values (the guard's decision logic);
- sample level: the REAL `wall_minus_mono` runs, with `$EPOCHREALTIME` stubbed
  (advanced by the `sleep` stub, 2.137 s per poll) and the `read` builtin (its
  /proc/uptime read) replaced by a stub deriving the uptime from it, so a stall
  between the two clock reads is injected exactly where it happens on the X5.

Locale: bash writes $EPOCHREALTIME with the locale's decimal point and mawk (the
Ubuntu 22.04 awk) parses split() fields with the locale's strtod, so the reader
and the guard are also run under a decimal-comma locale (skipped when the host
has none). The stubs themselves use LC_ALL=C awk so they are locale-proof.
"""
import functools
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "x5" / "go2_camera.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None or shutil.which("awk") is None, reason="needs bash and awk")

BASE = 1790700000.456          # a wall-minus-mono value in 2026
W0 = "1790712345.678901"       # fixed $EPOCHREALTIME for the sample-level stubs
POLLS = 15
STUB_DATE = "2026-09-30 12:00:00"
# deterministic reader noise within +-8 ms (the /proc/uptime quantum is 10 ms)
NOISE = [0.004, -0.007, 0.008, -0.003, 0.0, 0.006, -0.008, 0.002, -0.005, 0.007,
         -0.001, 0.003, -0.006, 0.005, -0.004, 0.001]


@functools.lru_cache(maxsize=None)
def _comma_locale():
    """Name of an installed locale whose decimal point is ',' (en_DK.utf8, de_DE.utf8, ...), or None."""
    try:
        names = subprocess.run(["locale", "-a"], capture_output=True, text=True, timeout=10).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    likely = [n for n in names if n.lower().startswith(("en_dk", "de_de", "fr_fr", "ru_ru", "es_es", "it_it"))]
    for name in likely or names:
        try:
            r = subprocess.run(["locale", "decimal_point"], capture_output=True, text=True, timeout=10,
                               env=dict(os.environ, LC_ALL=name))
        except (OSError, subprocess.SubprocessError):
            return None
        if r.stdout.strip() == ",":
            return name
    return None


def _locale_env(which):
    """Subprocess env for locale `which` ('default' = inherited, 'comma' = a decimal-comma locale)."""
    if which == "default":
        return None
    loc = _comma_locale()
    if loc is None:
        pytest.skip("no decimal-comma locale installed (locale -a)")
    return dict(os.environ, LC_ALL=loc)


LOCALES = ["default", "comma"]


def _src():
    return SCRIPT.read_text()


def _extract(name):
    """Text of the shell function `name` (one-liner or multi-line up to a column-0 '}')."""
    lines = _src().splitlines()
    for i, line in enumerate(lines):
        if re.match(rf"^{name}\(\)\s*\{{", line):
            if line.rstrip().endswith("}") and line.count("{") == line.count("}"):
                return line
            for j in range(i + 1, len(lines)):
                if lines[j] == "}":
                    return "\n".join(lines[i:j + 1])
    raise AssertionError(f"function {name} not found in {SCRIPT}")


def _default(name):
    """Value assigned to `name` in the defaults block (before camera.conf is sourced), or None."""
    head = _src().split('source "$CAMERA_CONF"')[0]
    m = re.search(rf"^{name}=(\S+)", head, re.M)
    return m.group(1) if m else None


def _bash_array(name, values):
    return f"{name}=(" + " ".join(f"{v:.6f}" for v in values) + ")"


def _run_guard(tmp_path, offsets, *, noise=None, stalls=None, sample_level=False, restart=1, polls=POLLS,
               thr=None, locale="default"):
    """Run the real clock_guard for `polls` polls. offsets[p] = clock step seen at poll p
    (p = 0 is the base read). Value level: wall_minus_mono = BASE + offsets[p] + noise[p].
    Sample level: the real wall_minus_mono, 5 reads per call; $EPOCHREALTIME = W0 + 2.137 s
    per poll (so its sub-second phase moves), written with the locale's decimal point;
    stalls[5 * p + i] delays the /proc/uptime read of sample i of poll p (a preemption
    between the two clocks). thr: CLOCK_STEP_THRESHOLD_S as camera.conf would set it
    (None = the script's default)."""
    env = _locale_env(locale)
    sep = "," if locale == "comma" else "."
    noise = noise if noise is not None else [0.0] * (polls + 1)
    stalls = stalls if stalls is not None else {}
    if thr is None:
        thr = _default("CLOCK_STEP_THRESHOLD_S")
        thr_line = f"CLOCK_STEP_THRESHOLD_S={thr}" if thr is not None else ": no CLOCK_STEP_THRESHOLD_S default"
    else:
        thr_line = f"CLOCK_STEP_THRESHOLD_S='{thr}'"
    stall_list = [stalls.get(k, 0.0) for k in range(5 * (polls + 1))]
    parts = [
        "set -uo pipefail",
        f"LOGDIR='{tmp_path}'",
        f"CLOCK_STEP_RESTART={restart}",
        thr_line,
        _extract("wall_minus_mono"),
        _extract("clock_guard"),
        f"BASE={BASE}; W0={W0}; SEP='{sep}'; POLL=0; RIDX=0; MAXP={polls}",
        _bash_array("OFF", offsets),
        _bash_array("NOISE", noise),
        _bash_array("STALL", stall_list),
        # stubs: one `sleep 2` per poll; the launch pid dies after MAXP polls
        "sleep() { POLL=$((POLL + 1)); set_wall; }",
        'kill() { if [ "$1" = "-0" ]; then [ "$POLL" -lt "$MAXP" ]; return; fi; command kill "$@"; }',
        'log() { echo "LOG: $*"; }',
        f"date() {{ echo '{STUB_DATE}'; }}",
        "clock_summary() { printf 'chrony stub'; }",
        'stop_pid_graceful() { echo "RESTART pid=$1 poll=$POLL"; }',
    ]
    if sample_level:
        parts += [
            # $EPOCHREALTIME of poll POLL, with the locale's decimal point as bash writes it
            "set_wall() { local w; w=$(LC_ALL=C awk -v w0=\"$W0\" -v p=\"$POLL\" "
            "'BEGIN { printf \"%.6f\", w0 + 2.137 * p }'); EPOCHREALTIME=${w/./$SEP}; }",
            "unset EPOCHREALTIME; set_wall",
            # `read -r u _ < /proc/uptime` inside wall_minus_mono (a subshell per call, so RIDX
            # counts 0..4 per call): u = wall - BASE - step + stall
            'read() { local v w=${EPOCHREALTIME/,/.}; v=$(LC_ALL=C awk -v w="$w" -v b="$BASE" -v o="${OFF[$POLL]}" '
            '-v st="${STALL[$((POLL * 5 + RIDX))]}" '
            "'BEGIN { printf \"%.6f\", w - b - o + st }'); RIDX=$((RIDX + 1)); printf -v \"$2\" '%s' \"$v\"; }",
        ]
    else:
        parts += [
            "set_wall() { :; }",
            'wall_minus_mono() { LC_ALL=C awk -v b="$BASE" -v o="${OFF[$POLL]}" -v n="${NOISE[$POLL]}" '
            "'BEGIN { printf \"%.3f\", b + o + n }'; }",
        ]
    parts += ["clock_guard 4242", 'echo "END poll=$POLL"']
    r = subprocess.run([BASH, "-c", "\n".join(parts)], capture_output=True, text=True, timeout=60, env=env)
    clog = tmp_path / "clock.log"
    return r, (clog.read_text() if clog.exists() else "")


def _step(size, at=5, polls=POLLS):
    return [0.0 if p < at else size for p in range(polls + 1)]


def _restart_poll(out):
    m = re.search(r"^RESTART pid=4242 poll=(\d+)$", out, re.M)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------ configuration

def test_script_parses():
    r = subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_threshold_default_below_chrony_makestep_and_overridable():
    assert _default("CLOCK_STEP_THRESHOLD_S") == "0.05"
    src = _src()
    # defined in the defaults block, i.e. before camera.conf is sourced (overridable there)
    assert src.index("CLOCK_STEP_THRESHOLD_S=0.05") < src.index('source "$CAMERA_CONF"')
    header = src.split("set -uo pipefail")[0]
    assert "CLOCK_STEP_THRESHOLD_S" in header
    assert "> 0.5 s" not in header


def test_help_prints_whole_header():
    """`go2_camera.sh` without a subcommand prints lines 2..N: exactly the header comment."""
    lines = _src().splitlines()
    m = re.search(r"sed -n '2,(\d+)p' \"\$0\"", _src())
    assert m, "help line not found"
    last = int(m.group(1))
    assert all(ln.startswith("#") for ln in lines[1:last]), "help range runs past the header"
    assert not lines[last].startswith("#"), "help range stops inside the header"


# ------------------------------------------------------------------ value level (guard logic)

@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("size", [0.101, -0.101, 0.3, -0.3, 0.48, 0.832])
def test_step_restarts_camera(tmp_path, size, locale):
    r, clog = _run_guard(tmp_path, _step(size), noise=NOISE, locale=locale)
    assert r.returncode == 0, r.stderr
    poll = _restart_poll(r.stdout)
    assert poll is not None, f"step {size} s not caught:\n{r.stdout}{r.stderr}"
    assert poll == 6, f"caught at poll {poll}"   # step from poll 5, 2 consecutive polls: poll 6
    m = re.search(r"^LOG: clock stepped by (\d+\.\d{3})s \(now " + STUB_DATE + r"; chrony stub\)$", r.stdout, re.M)
    assert m and abs(float(m.group(1)) - abs(size)) < 0.02
    assert "LOG: restarting realsense2_camera so frame stamps follow the new clock" in r.stdout
    assert re.fullmatch(STUB_DATE + r" STEP \d+\.\d{3}s chrony stub\n", clog)
    assert "WARNING" not in r.stdout


@pytest.mark.parametrize("locale", LOCALES)
def test_reader_noise_does_not_restart(tmp_path, locale):
    r, clog = _run_guard(tmp_path, _step(0.0), noise=NOISE, locale=locale)
    assert r.returncode == 0, r.stderr
    assert _restart_poll(r.stdout) is None and "STEP" not in clog
    assert f"END poll={POLLS}" in r.stdout


# A camera.conf typo must not turn the guard into a restart loop (t + 0 = 0 in awk
# restarted on the reader noise at poll 2-3): fall back to 0.05 s and say so.
@pytest.mark.parametrize("bad", ["0,05", "", "0", "0.000", "abc", "-0.1", "0.05s"])
def test_bad_threshold_falls_back_to_default(tmp_path, bad):
    r, clog = _run_guard(tmp_path, _step(0.0), noise=NOISE, thr=bad)
    assert r.returncode == 0, r.stderr
    assert _restart_poll(r.stdout) is None, r.stdout
    assert "STEP" not in clog
    assert f"END poll={POLLS}" in r.stdout
    warn = [ln for ln in r.stdout.splitlines() if "WARNING" in ln]
    assert len(warn) == 1 and "CLOCK_STEP_THRESHOLD_S" in warn[0] and "0.05" in warn[0], r.stdout
    # the fallback still catches a real step just over chrony's makestep
    r, _ = _run_guard(tmp_path, _step(0.101), noise=NOISE, thr=bad)
    assert _restart_poll(r.stdout) == 6, r.stdout


def test_threshold_override_is_honoured(tmp_path):
    r, _ = _run_guard(tmp_path, _step(0.101), noise=NOISE, thr="0.2")
    assert _restart_poll(r.stdout) is None and "WARNING" not in r.stdout, r.stdout
    r, _ = _run_guard(tmp_path, _step(0.3), noise=NOISE, thr="0.2")
    assert _restart_poll(r.stdout) == 6 and "WARNING" not in r.stdout, r.stdout


@pytest.mark.parametrize("stall", [-0.060, 0.060, -0.2])
def test_single_stalled_poll_does_not_restart(tmp_path, stall):
    """One poll off (a stall) and back: a real step persists, a stall does not."""
    off = [0.0] * (POLLS + 1)
    off[5] = stall
    r, clog = _run_guard(tmp_path, off, noise=NOISE)
    assert r.returncode == 0, r.stderr
    assert _restart_poll(r.stdout) is None, r.stdout
    assert "STEP" not in clog


def test_step_without_restart_logs_once_and_rebases(tmp_path):
    r, clog = _run_guard(tmp_path, _step(0.3), noise=NOISE, restart=0)
    assert r.returncode == 0, r.stderr
    assert _restart_poll(r.stdout) is None
    assert r.stdout.count("LOG: clock stepped by") == 1          # rebased: the new level is quiet
    assert len(re.findall(r" STEP \d+\.\d{3}s ", clog)) == 1
    assert f"END poll={POLLS}" in r.stdout


# ------------------------------------------------------------------ sample level (real reader)

@pytest.mark.parametrize("stalls", [
    {0: 0.060},                    # base taken with its first read stalled 60 ms
    {2: 0.060},                    # base taken with its middle read stalled 60 ms
    {0: 0.060, 4: 0.090},          # two of the five base reads stalled
    {5 * 5: 0.060},                # one stalled read in poll 5
    {5 * 5: 0.080, 6 * 5: 0.070},  # a stalled read in each of two consecutive polls
], ids=["base_first", "base_middle", "base_two", "poll5_one", "poll5_poll6"])
@pytest.mark.parametrize("locale", LOCALES)
def test_stalled_reads_do_not_restart(tmp_path, stalls, locale):
    r, clog = _run_guard(tmp_path, _step(0.0), stalls=stalls, sample_level=True, locale=locale)
    assert r.returncode == 0, r.stderr
    assert _restart_poll(r.stdout) is None, r.stdout
    assert "STEP" not in clog


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("size", [0.101, -0.3])
def test_step_restarts_with_real_reader(tmp_path, size, locale):
    r, _ = _run_guard(tmp_path, _step(size), stalls={2: 0.060}, sample_level=True, locale=locale)
    assert r.returncode == 0, r.stderr
    assert _restart_poll(r.stdout) == 6, r.stdout
    m = re.search(r"^LOG: clock stepped by (\d+\.\d{3})s ", r.stdout, re.M)
    assert m and abs(float(m.group(1)) - abs(size)) < 0.005, r.stdout


def _call_wmm(prelude, locale="default"):
    script = "\n".join(["set -uo pipefail", _extract("wall_minus_mono"), prelude, "wall_minus_mono"])
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=30,
                          env=_locale_env(locale))


@pytest.mark.parametrize("locale", LOCALES)
def test_wall_minus_mono_decimal_comma_and_median(locale):
    """$EPOCHREALTIME with a locale decimal comma; one outlier sample is ignored. Under a
    real decimal-comma locale mawk used to parse the split() fields with the locale's
    strtod and truncate both clocks to whole seconds ('1790699900,000')."""
    ups = ["100.000000", "100.000000", "100.090000", "100.000000", "100.000000"]
    prelude = "\n".join([
        "unset EPOCHREALTIME; EPOCHREALTIME='1790700000,456789'",
        "UPS=(" + " ".join(ups) + "); RIDX=0",
        "read() { printf -v \"$2\" '%s' \"${UPS[$RIDX]}\"; RIDX=$((RIDX + 1)); }",
    ])
    r = _call_wmm(prelude, locale)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "1790699900.457"


def test_wall_minus_mono_forks_nothing_between_clock_reads(tmp_path):
    """Neither `date` nor `cut` may run: a fork between the two clock reads is the noise
    (up to 21-24 ms measured under load) that a 50 ms threshold cannot afford."""
    if int(subprocess.run([BASH, "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True).stdout) < 5:
        pytest.skip("EPOCHREALTIME needs bash >= 5")
    calls = tmp_path / "calls"
    prelude = f"date() {{ echo date >> '{calls}'; command date \"$@\"; }}\ncut() {{ echo cut >> '{calls}'; command cut \"$@\"; }}"
    r = _call_wmm(prelude)
    assert r.returncode == 0, r.stderr
    assert not calls.exists(), calls.read_text()


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("prelude", [":", "unset EPOCHREALTIME"], ids=["bash5", "no_EPOCHREALTIME"])
def test_wall_minus_mono_matches_real_clocks(prelude, locale):
    """Real clocks; without $EPOCHREALTIME (bash < 5) it must fall back to `date`, not die on
    `set -u` (an empty reading would look like a 1.8e9 s step on every poll: a restart loop).
    Under a decimal-comma locale the output must still be the same '.' number."""
    r = _call_wmm(prelude, locale)
    assert r.returncode == 0, r.stderr
    assert re.fullmatch(r"\d+\.\d{3}", r.stdout), r.stdout
    ref = time.time() - float(Path("/proc/uptime").read_text().split()[0])
    assert abs(float(r.stdout) - ref) < 0.05


# ------------------------------------------------------ clock_synced (start gate)
@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("offset, synced", [("-0.500000", False), ("0.500000", False),
                                            ("-0.005000", True), ("0.005000", True), ("0.011000", False)])
def test_clock_synced_uses_abs_offset_in_any_locale(tmp_path, offset, synced, locale):
    """clock_synced gates the first camera start on |chrony offset| <= CLOCK_MAX_OFFSET_S (0.010).
    Under a decimal-comma locale mawk parsed '-0.500000' as -0 and let a 0.5 s offset through."""
    env = _locale_env(locale)
    bindir = tmp_path / "bin"; bindir.mkdir()
    fake = bindir / "chronyc"
    fake.write_text("#!/bin/bash\necho 'C000020A,192.0.2.10,4,1790000000.0,%s,0,0,0,0,0,0,0,2.0,Normal'\n" % offset)
    fake.chmod(0o755)
    prog = "\n".join(["CLOCK_MAX_OFFSET_S=0.010", _extract("have_chrony"), _extract("clock_synced"),
                      "clock_synced && echo SYNCED || echo NOT_SYNCED"])
    env = dict(env or os.environ); env["PATH"] = f"{bindir}:{env.get('PATH', '')}"
    out = subprocess.run([BASH, "-c", prog], capture_output=True, text=True, timeout=20, env=env).stdout.strip()
    assert out == ("SYNCED" if synced else "NOT_SYNCED"), (offset, locale, out)
