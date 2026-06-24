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
    common.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value with dotted keys, matching scripts/train.py.",
    )
    common.add_argument(
        "--require-external-exists",
        action="store_true",
        help="Fail before probing if configured external repo directories are missing.",
    )

    parser = argparse.ArgumentParser(
        description="Static and compute-node probes for WAM-Policy action-stream FFM runtime binding.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe-static-source", parents=[common])
    subparsers.add_parser("probe-import", parents=[common])
    subparsers.add_parser("probe-load-model", parents=[common])
    subparsers.add_parser("probe-manifest-one-sample", parents=[common])
    forward = subparsers.add_parser("probe-action-forward", parents=[common])
    forward.add_argument("--s-src", type=float, default=0.1, help="Probe source flow time.")
    forward.add_argument("--s-tgt", type=float, default=0.9, help="Probe target flow time.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg, external_paths = _load_config_and_bootstrap(args)

    _print_header("Config")
    print(f"repo_root: {REPO_ROOT}")
    print(f"config: {Path(args.config).resolve()}")
    for path in external_paths:
        print(f"external_repo: {path} exists={path.is_dir()}")

    if args.command == "probe-static-source":
        probe_static_source(cfg)
        return 0
    if args.command == "probe-import":
        probe_imports()
        return 0
    if args.command == "probe-load-model":
        probe_load_model(cfg)
        return 0
    if args.command == "probe-manifest-one-sample":
        probe_manifest_one_sample(cfg)
        return 0
    if args.command == "probe-action-forward":
        probe_action_forward(cfg, s_src=args.s_src, s_tgt=args.s_tgt)
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


def probe_static_source(cfg: Mapping[str, Any]) -> None:
    lingbot_root = _configured_lingbot_root(cfg)
    source_paths = {
        "lingbot_modules_utils": lingbot_root / "wan_va" / "modules" / "utils.py",
        "lingbot_modules_model": lingbot_root / "wan_va" / "modules" / "model.py",
        "lingbot_train": lingbot_root / "wan_va" / "train.py",
        "wam_adapter": REPO_ROOT / "src" / "wam_policy" / "adapters" / "lingbot_va.py",
        "wam_student": REPO_ROOT / "src" / "wam_policy" / "models" / "lingbot_student.py",
        "wam_action_dataset": REPO_ROOT / "src" / "wam_policy" / "data" / "action_dataset.py",
    }

    _print_header("Static Source Files")
    for name, path in source_paths.items():
        print(f"{name}: {path} exists={path.is_file()}")

    adapter_source = _read_text_if_exists(source_paths["wam_adapter"])
    student_source = _read_text_if_exists(source_paths["wam_student"])
    dataset_source = _read_text_if_exists(source_paths["wam_action_dataset"])
    model_source = _read_text_if_exists(source_paths["lingbot_modules_model"])
    train_source = _read_text_if_exists(source_paths["lingbot_train"])

    _print_header("WAM Action Adapter Alignment")
    _print_check("adapter exposes build_action_context", "def build_action_context" in adapter_source)
    _print_check("adapter exposes attach_action_time_pair_conditioner", "def attach_action_time_pair_conditioner" in adapter_source)
    _print_check("adapter exposes install_action_flowmap_hook", "def install_action_flowmap_hook" in adapter_source)
    _print_check("adapter exposes forward_action_flowmap", "def forward_action_flowmap" in adapter_source)
    _print_check("adapter uses mode-specific action conditioner", "wam_action_time_pair_conditioner" in adapter_source)
    _print_check("adapter keeps video conditioner alias", "wam_time_pair_conditioner" in adapter_source)
    _print_check("student exposes LingBotActionFlowMapStudent", "class LingBotActionFlowMapStudent" in student_source)
    _print_check("action dataset exposes ActionChunkManifestDataset", "class ActionChunkManifestDataset" in dataset_source)

    _print_header("LingBot Action Source Hints")
    _print_check("model source mentions action_mode", "action_mode" in model_source)
    _print_check("train source mentions action_dict", "action_dict" in train_source)
    print("action forward binding still requires server-side probe against the real LingBot action contract.")


def probe_imports() -> None:
    _print_header("Dependency Status")
    for module_name in ("torch", "diffusers", "transformers", "easydict", "numpy", "yaml"):
        print(f"{module_name}: {_module_status(module_name)}")

    _print_header("LingBot Runtime")
    wan_va = _import_required("wan_va")
    print(f"wan_va.__file__: {getattr(wan_va, '__file__', None)}")
    utils = _import_required("wan_va.modules.utils")
    print(f"wan_va.modules.utils.__file__: {getattr(utils, '__file__', None)}")
    load_transformer = getattr(utils, "load_transformer", None)
    if load_transformer is None:
        raise AttributeError("wan_va.modules.utils.load_transformer was not found.")
    print(f"load_transformer_signature: {_safe_signature(load_transformer)}")


def probe_load_model(cfg: Mapping[str, Any]) -> None:
    from wam_policy.adapters.lingbot_va import LingBotVAAdapter

    _print_header("Load LingBot Model")
    adapter = LingBotVAAdapter(cfg)
    print(f"checkpoint_path: {adapter.checkpoint_path}")
    print(f"device: {adapter.device}")
    print(f"torch_dtype: {adapter.torch_dtype}")
    model = adapter.load_pretrained_video_action_model()
    _describe_model(model)


def probe_manifest_one_sample(cfg: Mapping[str, Any]) -> None:
    from wam_policy.data.action_dataset import ActionChunkManifestDataset

    _print_header("Action Manifest One Sample")
    dataset = ActionChunkManifestDataset.from_config(cfg)
    if len(dataset) <= 0:
        raise ValueError("Configured action manifest is empty.")
    sample = dataset[0]
    print(f"manifest_path: {dataset.manifest_path}")
    print(f"dataset_len: {len(dataset)}")
    print(f"sample_id: {sample.get('meta', {}).get('sample_id')}")
    print(f"a_clean: {_tensor_summary(sample['a_clean'])}")
    print(f"context_type: {type(sample.get('context')).__name__}")
    print(f"z_condition: {_tensor_summary(sample.get('z_condition'))}")
    print(f"prompt: {sample.get('prompt', '')!r}")


def probe_action_forward(cfg: Mapping[str, Any], s_src: float, s_tgt: float) -> None:
    if not s_src < s_tgt:
        raise ValueError("--s-src must be smaller than --s-tgt.")

    import torch

    from wam_policy.adapters.anyflow import build_action_flowmap_pair
    from wam_policy.adapters.lingbot_va import LingBotVAAdapter
    from wam_policy.data.action_dataset import ActionChunkManifestDataset
    from wam_policy.models.flow_map_heads import FlowTimePairEmbedding

    _print_header("Load LingBot Model")
    adapter = LingBotVAAdapter(cfg)
    model = adapter.load_pretrained_video_action_model()
    _describe_model(model)

    _print_header("Attach Action Flow-Map Conditioner")
    conditioner = FlowTimePairEmbedding(
        hidden_dim=_model_int(cfg, "time_pair_hidden_dim", default=256),
        mlp_hidden_dim=_model_optional_int(cfg, "time_pair_mlp_hidden_dim"),
    )
    adapter.attach_action_time_pair_conditioner(model, conditioner)
    conditioner.to(device=adapter.device, dtype=adapter.torch_dtype)
    model.eval()
    print(f"conditioner_hidden_dim: {conditioner.hidden_dim}")
    print(f"conditioner_timestep_proj_dim: {conditioner.timestep_proj_dim}")

    _print_header("Load One Action Manifest Sample")
    dataset = ActionChunkManifestDataset.from_config(cfg)
    if len(dataset) <= 0:
        raise ValueError("Configured action manifest is empty.")
    sample = dataset[0]
    a_clean = _batched_sample_a_clean(sample["a_clean"]).to(
        device=adapter.device,
        dtype=adapter.torch_dtype,
    )
    z_condition = sample.get("z_condition")
    if isinstance(z_condition, torch.Tensor):
        z_condition = z_condition.unsqueeze(0).to(device=adapter.device, dtype=adapter.torch_dtype)
    s_src_tensor = torch.full((a_clean.shape[0],), s_src, device=adapter.device)
    s_tgt_tensor = torch.full((a_clean.shape[0],), s_tgt, device=adapter.device)
    flowmap_pair = build_action_flowmap_pair(
        a_clean=a_clean,
        s_src=s_src_tensor,
        s_tgt=s_tgt_tensor,
    )
    context = adapter.build_action_context({"context": [sample["context"]]})

    print(f"sample_id: {sample.get('meta', {}).get('sample_id')}")
    print(f"a_clean: {_tensor_summary(a_clean)}")
    print(f"a_src: {_tensor_summary(flowmap_pair.a_src)}")
    print(f"target_u: {_tensor_summary(flowmap_pair.target_u)}")
    print(f"z_condition: {_tensor_summary(z_condition)}")

    _print_header("LingBot Action Input Contract")
    input_dict = adapter.build_lingbot_action_input(
        model,
        flowmap_pair.a_src,
        flowmap_pair.s_src,
        flowmap_pair.s_tgt,
        context,
        z_condition=z_condition,
    )
    for key, value in input_dict.items():
        print(f"{key}: {_tensor_summary(value)}")

    _print_header("No-Grad Action Forward")
    with torch.no_grad():
        pred_u = adapter.forward_action_flowmap(
            model,
            flowmap_pair.a_src,
            flowmap_pair.s_src,
            flowmap_pair.s_tgt,
            context,
            z_condition=z_condition,
        )
    print(f"pred_u: {_tensor_summary(pred_u)}")
    print(f"pred_u_matches_a_src_shape: {tuple(pred_u.shape) == tuple(flowmap_pair.a_src.shape)}")
    print(f"pred_u_all_finite: {bool(torch.isfinite(pred_u).all().item())}")


def _batched_sample_a_clean(a_clean: Any) -> Any:
    import torch

    if not isinstance(a_clean, torch.Tensor):
        raise TypeError("a_clean must be a torch.Tensor.")
    batched = a_clean.unsqueeze(0)
    if batched.ndim in {2, 3, 4}:
        return batched
    raise ValueError("A single manifest a_clean sample must become [B, D], [B, T, D], or [B, C, T].")


def _describe_model(model: Any) -> None:
    print(f"model_class: {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"model_forward_signature: {_safe_signature(getattr(model, 'forward', None))}")
    print(f"patch_size: {getattr(model, 'patch_size', None)}")
    print(f"_time_embed_signature: {_safe_signature(getattr(model, '_time_embed', None))}")
    condition_embedder = getattr(model, "condition_embedder", None)
    time_proj = getattr(condition_embedder, "time_proj", None)
    print(f"condition_embedder: {condition_embedder.__class__.__name__ if condition_embedder else None}")
    print(f"time_proj_out_features: {getattr(time_proj, 'out_features', None)}")


def _configured_lingbot_root(cfg: Mapping[str, Any]) -> Path:
    paths_cfg = cfg.get("paths")
    raw_path = "external/lingbot-va"
    if isinstance(paths_cfg, Mapping):
        configured_path = paths_cfg.get("lingbot_repo")
        if isinstance(configured_path, str) and configured_path:
            raw_path = configured_path
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _read_text_if_exists(path: Path) -> str:
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
