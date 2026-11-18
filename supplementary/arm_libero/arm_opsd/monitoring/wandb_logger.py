"""Optional Weights & Biases logging of the training events; off unless configured.

Enable it with a ``wandb`` section in the site file (or the standard WANDB_* environment variables)::

    "wandb": {"project": "<project>", "entity": "<entity or null>", "mode": "online" | "offline"}

Rank 0 writes every event to ``rank-00/events.jsonl`` in any case; this logger only mirrors the
training curves (losses, held-out loss, timings, prediction videos) to W&B. A logging failure never
stops training.
"""
import os
import traceback


def event_metrics(row):
    """W&B row for one events.jsonl record, or None when the event has no curve."""
    event = row.get("event")
    if event == "round_complete":
        counters = row["counters"]
        metrics = {"optimizer_step": counters["student_updates"],
                   "train/contexts": counters["collection_contexts"],
                   "train/offline_contexts": counters["offline_contexts"],
                   "train/replay_contexts": counters["replay_contexts"],
                   "train/round_seconds": row["round_ms"] / 1000,
                   "train/collect_round": float(bool(row.get("collect_round", False)))}
        if row.get("student_grad_norm") is not None:
            metrics["train/student_grad_norm"] = row["student_grad_norm"]
        if row.get("student_lr") is not None:
            metrics["optim/student_lr"] = row["student_lr"]
        for key, value in (row.get("grad") or {}).items():
            metrics[f"train/loss/{key}"] = value
        metrics.update({k: v for k, v in row.items() if k.startswith(("adapter/", "adapter_ema/"))})
        metrics.update({f"train/{key}": value for key, value in row.get("global_losses", {}).items()})
        metrics.update({f"train/data/{key}": value for key, value in row.get("data", {}).items()})
        metrics.update({f"train/time/{key}": value for key, value in row.get("time", {}).items()})
        for task, values in row.get("task_totals", {}).items():
            metrics.update({f"train/task_{task}/{k}": v for k, v in values.items()})
        return metrics
    if event == "heldout":
        return {"optimizer_step": row["optimizer_step"], **{f"train/{k}": v for k, v in row["metrics"].items()}}
    if event == "cache_progress":
        return {"cache/round": row["round"], "cache/global_entries": row["global_entries"],
                "cache/requested": row["requested"], "cache/round_s": row.get("round_ms", 0.) / 1000}
    if event == "prediction_video":
        return {"optimizer_step": row["optimizer_step"], "visualization/success": float(row["success"]),
                "visualization/closed_loop": {"__video__": row["path"], "fps": row["fps"],
                                              "caption": f"task {row['task_id']} update {row['optimizer_step']}"}}
    return None


class WandbLogger:
    def __init__(self, settings, *, name, group, config, directory):
        import wandb
        self.wandb = wandb
        self.run = wandb.init(project=settings.get("project"), entity=settings.get("entity"),
                              mode=settings.get("mode", "online"), name=name, group=group, config=config,
                              dir=str(directory))
        self.run.define_metric("optimizer_step")
        for prefix in ("train/*", "optim/*", "adapter/*", "adapter_ema/*", "visualization/*"):
            self.run.define_metric(prefix, step_metric="optimizer_step")

    @classmethod
    def from_site(cls, settings, **kwargs):
        settings = dict(settings or {})
        settings.setdefault("project", os.environ.get("WANDB_PROJECT"))
        if not settings.get("project") or os.environ.get("WANDB_MODE") == "disabled":
            return None
        try:
            return cls(settings, **kwargs)
        except Exception:
            traceback.print_exc()
            return None

    def log(self, row):
        metrics = event_metrics(row)
        if metrics is None:
            return
        try:
            for key, value in list(metrics.items()):
                if isinstance(value, dict) and "__video__" in value:
                    metrics[key] = self.wandb.Video(value["__video__"], fps=value["fps"], caption=value["caption"],
                                                    format="mp4")
            self.run.log(metrics)
        except Exception:
            traceback.print_exc()

    def finish(self):
        try:
            self.run.finish()
        except Exception:
            traceback.print_exc()
