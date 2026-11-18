"""Bridge to the QLM-Bench training-environment CLI (``python -m qlm_bench.training_env``): training
profiles and the global episode schedule shared by all simulation workers of a run."""
import json
from pathlib import Path
import subprocess
import tempfile
from wam_policy.opsd.config import digest

QLM_ENTRY = "qlm_bench.training_env"


def command(config, operation, arguments, *, timeout=None):
    from wam_policy.opsd.launch import child_environment
    proc = subprocess.run([config.need("qlm_python"), "-m", QLM_ENTRY, operation, *map(str, arguments)],
                          env=child_environment(config, cpu=True), capture_output=True, text=True,
                          timeout=timeout or config.rpc_timeout_s)
    if proc.returncode:
        raise RuntimeError(f"QLM {operation} failed: " + proc.stderr[-3000:])
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def make_profile(config, path, *, tasks=None, max_active_per_task=None):
    """Write the QLM training profile (default: nine training tasks, two active episodes per task;
    M-OPD expert collection: only the expert tasks)."""
    args = ["--pool", config.need("train_pool"), "--profile", str(path)]
    if tasks:
        args += ["--tasks", ",".join(tasks), "--max-active-per-task", max_active_per_task]
    command(config, "make-profile", args)
    return str(path)


class TrainingControl:
    def __init__(self, config, output, profile):
        self.config, self.profile = config, str(profile)
        token = digest(str(Path(output).absolute()))[:16]
        self.segment = "train-" + token
        self.db = Path(output).absolute() / "control" / "episodes.sqlite"

    def call(self, operation, *extra):
        return command(self.config, operation, ["--profile", self.profile, "--db", self.db, *extra])

    def initialize(self, state=None):
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self.call("schedule-init", "--pool", self.config.need("train_pool"), "--segment", self.segment)
        if state is not None:
            with tempfile.TemporaryDirectory(prefix="schedule-restore-") as folder:
                p = Path(folder) / "snapshot.json"
                p.write_text(json.dumps(state, allow_nan=False))
                self.call("schedule-restore", "--segment", self.segment, "--input", p)

    def snapshot(self):
        return self.call("schedule-snapshot")

    def stop(self):
        return self.call("schedule-stop")
