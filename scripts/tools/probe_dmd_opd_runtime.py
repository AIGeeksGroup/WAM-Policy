#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    common.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    common.add_argument("--require-external-exists", action="store_true")

    parser = argparse.ArgumentParser(
        description="Static and compute-node probes for WAM-Policy DMD OPD runtime binding.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe-static-source", parents=[common])
    subparsers.add_parser("probe-import", parents=[common])
    subparsers.add_parser("probe-load-model", parents=[common])
    subparsers.add_parser("probe-rollout-manifest", parents=[common])
    forward = subparsers.add_parser("probe-dmd-forward", parents=[common])
    forward.add_argument("--s-src", type=float, default=0.1)
    forward.add_argument("--s-tgt", type=float, default=0.9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg, external_paths = _load_config_and_bootstrap(args)
    stage = _stage(cfg)

    _print_header("Config")
    print(f"repo_root: {REPO_ROOT}")
    print(f"config: {Path(args.config).resolve()}")
    print(f"stage: {stage}")
    for path in external_paths:
        print(f"external_repo: {path} exists={path.is_dir()}")

    if args.command == "probe-static-source":
        probe_static_source(stage)
        return 0
    if args.command == "probe-import":
        probe_imports()
        return 0
    if args.command == "probe-load-model":
        probe_load_model(cfg)
        return 0
    if args.command == "probe-rollout-manifest":
        probe_rollout_manifest(cfg)
        return 0
    if args.command == "probe-dmd-forward":
        probe_dmd_forward(cfg, stage=stage, s_src=args.s_src, s_tgt=args.s_tgt)
        return 0
    raise ValueError(f"Unknown command: {args.command}")


def _load_config_and_bootstrap(args: argparse.Namespace) -> tuple[dict[str, Any], list[Path]]:
    from wam_policy.utils.config import apply_overrides, load_config
    from wam_policy.utils.external import bootstrap_external_repos

    cfg = apply_overrides(load_config(args.config), args.overrides)
    external_paths = bootstrap_external_repos(
        cfg,
        require_exists=bool(args.require_external_exists),
    )
    return cfg, external_paths


def probe_static_source(stage: str) -> None:
    trainer_source = _read_text(REPO_ROOT / "src" / "wam_policy" / "opd" / "trainer.py")
    adapter_source = _read_text(REPO_ROOT / "src" / "wam_policy" / "adapters" / "lingbot_va.py")
    dataset_source = _read_text(
        REPO_ROOT / "src" / "wam_policy" / "data" / "rollout_manifest_dataset.py"
    )
    losses_source = _read_text(REPO_ROOT / "src" / "wam_policy" / "opd" / "losses.py")

    _print_header("WAM DMD OPD Static Alignment")
    _print_check("rollout manifest dataset exists", "class RolloutManifestDataset" in dataset_source)
    _print_check("DMD surrogate loss exists", "def dmd_score_surrogate_loss" in losses_source)
    _print_check("video DMD trainer exists", "class VideoDMDOPDTrainer" in trainer_source)
    _print_check(
        "video teacher score placeholder exists",
        "def teacher_video_score_or_velocity" in adapter_source,
    )
    if stage == "action_dmd":
        _print_check("action DMD trainer exists", "class ActionDMDOPDTrainer" in trainer_source)
    print("Real teacher score/velocity binding is validated only in probe-dmd-forward.")


def probe_imports() -> None:
    _print_header("Dependency Status")
    for module_name in ("torch", "diffusers", "transformers", "easydict", "numpy", "yaml"):
        print(f"{module_name}: {_module_status(module_name)}")
    _print_header("LingBot Runtime")
    wan_va = _import_required("wan_va")
    print(f"wan_va.__file__: {getattr(wan_va, '__file__', None)}")
    utils = _import_required("wan_va.modules.utils")
    print(f"wan_va.modules.utils.__file__: {getattr(utils, '__file__', None)}")


def probe_load_model(cfg: Mapping[str, Any]) -> None:
    from wam_policy.adapters.lingbot_va import LingBotVAAdapter

    _print_header("Load LingBot Model")
    adapter = LingBotVAAdapter(cfg)
    print(f"checkpoint_path: {adapter.checkpoint_path}")
    print(f"device: {adapter.device}")
    print(f"torch_dtype: {adapter.torch_dtype}")
    model = adapter.load_pretrained_video_action_model()
    print(f"model_class: {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"model_forward_signature: {_safe_signature(getattr(model, 'forward', None))}")


def probe_rollout_manifest(cfg: Mapping[str, Any]) -> None:
    from wam_policy.data.rollout_manifest_dataset import RolloutManifestDataset

    _print_header("Rollout Manifest One Sample")
    dataset = RolloutManifestDataset.from_config(cfg)
    if len(dataset) <= 0:
        raise ValueError("Configured rollout manifest is empty.")
    sample = dataset[0]
    print(f"manifest_path: {dataset.manifest_path}")
    print(f"dataset_len: {len(dataset)}")
    print(f"sample_id: {sample['meta'].get('sample_id')}")
    print(f"z_student: {_tensor_summary(sample.get('z_student'))}")
    print(f"a_student: {_tensor_summary(sample.get('a_student'))}")
    print(f"z_condition: {_tensor_summary(sample.get('z_condition'))}")


def probe_dmd_forward(cfg: Mapping[str, Any], stage: str, s_src: float, s_tgt: float) -> None:
    if stage != "video_dmd":
        raise NotImplementedError("Action DMD forward probe is added with ActionDMDOPDTrainer.")
    if not s_src < s_tgt:
        raise ValueError("--s-src must be smaller than --s-tgt.")

    import torch

    from wam_policy.adapters.anyflow import build_video_flowmap_pair
    from wam_policy.adapters.lingbot_va import LingBotVAAdapter
    from wam_policy.data.rollout_manifest_dataset import RolloutManifestDataset
    from wam_policy.models.flow_map_heads import FlowTimePairEmbedding

    adapter = LingBotVAAdapter(cfg)
    model = adapter.load_pretrained_video_action_model()
    conditioner = FlowTimePairEmbedding(
        hidden_dim=_model_int(cfg, "time_pair_hidden_dim", default=256),
        mlp_hidden_dim=_model_optional_int(cfg, "time_pair_mlp_hidden_dim"),
    )
    adapter.attach_video_time_pair_conditioner(model, conditioner)
    conditioner.to(device=adapter.device, dtype=adapter.torch_dtype)
    model.eval()

    dataset = RolloutManifestDataset.from_config(cfg)
    sample = dataset[0]
    z_student = sample["z_student"]
    if not isinstance(z_student, torch.Tensor):
        raise ValueError("Video DMD forward probe requires z_student tensor.")
    z_student = z_student.unsqueeze(0).to(device=adapter.device, dtype=adapter.torch_dtype)
    s_src_tensor = torch.full((z_student.shape[0],), s_src, device=adapter.device)
    s_tgt_tensor = torch.full((z_student.shape[0],), s_tgt, device=adapter.device)
    pair = build_video_flowmap_pair(z_student, s_src_tensor, s_tgt_tensor)
    context = adapter.build_video_context({"context": [sample["context"]]})

    with torch.no_grad():
        pred_u = adapter.forward_video_flowmap(model, pair.z_src, pair.s_src, pair.s_tgt, context)
        teacher_score = adapter.teacher_video_score_or_velocity(
            z_state=pair.z_src,
            s_src=pair.s_src,
            s_tgt=pair.s_tgt,
            context=context,
        )
    print(f"pred_u: {_tensor_summary(pred_u)}")
    print(f"teacher_score: {_tensor_summary(teacher_score)}")


def _stage(cfg: Mapping[str, Any]) -> str:
    experiment = cfg.get("experiment")
    if not isinstance(experiment, Mapping) or not isinstance(experiment.get("stage"), str):
        raise ValueError("Config experiment.stage must be set.")
    return experiment["stage"]


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _tensor_summary(value: Any) -> str:
    if not hasattr(value, "shape"):
        return repr(value)
    return (
        f"shape={tuple(value.shape)} "
        f"dtype={getattr(value, 'dtype', None)} "
        f"device={getattr(value, 'device', None)}"
    )


def _module_status(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return "missing"
    origin = spec.origin or "namespace"
    return f"available origin={origin}"


def _print_check(label: str, passed: bool) -> None:
    status = "OK" if passed else "MISSING"
    print(f"{status}: {label}")


def _import_required(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to import required module {module_name}: {exc}") from exc


def _safe_signature(obj: Any) -> str:
    if obj is None:
        return "None"
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def _model_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    value = default
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, Mapping) and key in model_cfg:
        value = model_cfg[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"model.{key} must be an int.")
    return value


def _model_optional_int(cfg: Mapping[str, Any], key: str) -> int | None:
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, Mapping) or key not in model_cfg:
        return None
    value = model_cfg[key]
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"model.{key} must be an int when provided.")
    return value


def _print_header(title: str) -> None:
    print(f"\n== {title} ==")


if __name__ == "__main__":
    raise SystemExit(main())
