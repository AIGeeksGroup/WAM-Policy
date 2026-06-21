#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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
        description="Static and compute-node probes for WAM-Policy video-stream FFM runtime binding.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "probe-static-source",
        parents=[common],
        help="Read LingBot/WAM source files and check adapter alignment without importing torch.",
    )
    subparsers.add_parser(
        "probe-import",
        parents=[common],
        help="Check external repo import paths and key LingBot runtime symbols.",
    )
    subparsers.add_parser(
        "probe-load-model",
        parents=[common],
        help="Load the configured LingBot checkpoint and print model binding metadata.",
    )
    subparsers.add_parser(
        "probe-manifest-one-sample",
        parents=[common],
        help="Load one manifest sample and validate z_clean/text_emb shapes.",
    )
    forward = subparsers.add_parser(
        "probe-video-forward",
        parents=[common],
        help="Load one sample and run a no-grad LingBot video flow-map forward probe.",
    )
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
    if args.command == "probe-video-forward":
        probe_video_forward(cfg, s_src=args.s_src, s_tgt=args.s_tgt)
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
        "lingbot_server": lingbot_root / "wan_va" / "wan_va_server.py",
        "lingbot_train": lingbot_root / "wan_va" / "train.py",
        "wam_adapter": REPO_ROOT / "src" / "wam_policy" / "adapters" / "lingbot_va.py",
        "wam_runtime": REPO_ROOT / "src" / "wam_policy" / "utils" / "runtime.py",
    }

    _print_header("Static Source Files")
    for name, path in source_paths.items():
        print(f"{name}: {path} exists={path.is_file()}")

    utils_source = _read_text(source_paths["lingbot_modules_utils"])
    model_source = _read_text(source_paths["lingbot_modules_model"])
    server_source = _read_text(source_paths["lingbot_server"])
    train_source = _read_text(source_paths["lingbot_train"])
    adapter_source = _read_text(source_paths["wam_adapter"])
    runtime_source = _read_text(source_paths["wam_runtime"])

    _print_header("LingBot Source Signatures")
    print(f"load_transformer: {_format_function_signature(utils_source, 'load_transformer')}")
    print(
        "WanTransformer3DModel._time_embed: "
        f"{_format_method_signature(model_source, 'WanTransformer3DModel', '_time_embed')}"
    )
    print(
        "WanTransformer3DModel.forward: "
        f"{_format_method_signature(model_source, 'WanTransformer3DModel', 'forward')}"
    )
    lingbot_utils_source = _read_text(lingbot_root / "wan_va" / "utils" / "utils.py")
    print(f"get_mesh_id: {_format_function_signature(lingbot_utils_source, 'get_mesh_id')}")
    print(
        "data_seq_to_patch: "
        f"{_format_function_signature(lingbot_utils_source, 'data_seq_to_patch')}"
    )

    _print_header("LingBot Static Facts")
    _print_check(
        "load_transformer accepts **kwargs",
        "def load_transformer" in utils_source and "**kwargs" in utils_source,
    )
    _print_check(
        "model forward accepts update_cache/cache_name/action_mode/train_mode",
        all(token in model_source for token in ("update_cache=0", 'cache_name="pos"', "action_mode=False", "train_mode=False")),
    )
    _print_check(
        "_time_embed unflattens timestep_proj to [B,L,6,C]",
        "timestep_proj.unflatten(2, (6, -1))" in model_source,
    )
    _print_check(
        "server inference uses single-stream latent_res_lst forward",
        "input_dict['latent_res_lst']" in server_source and "action_mode=False" in server_source,
    )
    _print_check(
        "train path uses dual-head latent_dict/action_dict forward_train",
        "input_dict['latent_dict']" in train_source and "input_dict['action_dict']" in train_source,
    )

    _print_header("WAM Adapter Alignment")
    _print_check("runtime supports rocm/hip aliases", "rocm" in runtime_source and "hip" in runtime_source)
    _print_check("runtime defaults LingBot attn_mode to torch", "lingbot_attn_mode" in runtime_source and '"torch"' in runtime_source)
    _print_check("adapter passes attn_mode to load_transformer", "attn_mode=self.lingbot_attn_mode" in adapter_source)
    _print_check("adapter installs flash-attn stub for torch attn mode", "_install_flash_attention_stub_if_missing" in adapter_source)
    _print_check("adapter calls single-stream update_cache/cache_name forward", "update_cache=0" in adapter_source and "cache_name=self._cache_name()" in adapter_source)
    _print_check("adapter patches condition_embedder.forward for real inference path", "_patch_condition_embedder_forward" in adapter_source)
    _print_check("adapter retains _time_embed fallback for smoke tests/train path", "_wam_original_time_embed" in adapter_source)


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
    print(f"load_transformer: {load_transformer}")
    print(f"load_transformer_signature: {_safe_signature(load_transformer)}")

    _print_header("AnyFlow Runtime")
    print(f"far: {_module_status('far')}")
    print("AnyFlow external runtime is not imported by the current video_ffm path.")


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
    import torch

    from wam_policy.adapters.lingbot_va import LingBotVAAdapter
    from wam_policy.data.video_latent_dataset import VideoLatentManifestDataset

    _print_header("Manifest One Sample")
    dataset = VideoLatentManifestDataset.from_config(cfg)
    if len(dataset) <= 0:
        raise ValueError("Configured video latent manifest is empty.")

    sample = dataset[0]
    z_clean = _batched_sample_z_clean(sample["z_clean"])
    adapter = LingBotVAAdapter(cfg)
    text_emb = adapter.normalize_video_context(
        [sample["context"]],
        batch_size=z_clean.shape[0],
        device=torch.device("cpu"),
        dtype=z_clean.dtype,
    )

    meta = sample.get("meta", {})
    print(f"manifest_path: {dataset.manifest_path}")
    print(f"dataset_len: {len(dataset)}")
    print(f"sample_id: {meta.get('sample_id')}")
    print(f"z_clean: {_tensor_summary(z_clean)}")
    print(f"text_emb: {_tensor_summary(text_emb)}")
    print(f"prompt: {sample.get('prompt', '')!r}")


def probe_video_forward(cfg: Mapping[str, Any], s_src: float, s_tgt: float) -> None:
    if not s_src < s_tgt:
        raise ValueError("--s-src must be smaller than --s-tgt.")

    import torch

    from wam_policy.adapters.anyflow import build_video_flowmap_pair
    from wam_policy.adapters.lingbot_va import LingBotVAAdapter
    from wam_policy.data.video_latent_dataset import VideoLatentManifestDataset
    from wam_policy.models.flow_map_heads import FlowTimePairEmbedding

    _print_header("Load LingBot Model")
    adapter = LingBotVAAdapter(cfg)
    model = adapter.load_pretrained_video_action_model()
    _describe_model(model)

    _print_header("Attach Video Flow-Map Conditioner")
    conditioner = FlowTimePairEmbedding(
        hidden_dim=_model_int(cfg, "time_pair_hidden_dim", default=256),
        mlp_hidden_dim=_model_optional_int(cfg, "time_pair_mlp_hidden_dim"),
    )
    adapter.attach_video_time_pair_conditioner(model, conditioner)
    conditioner.to(device=adapter.device, dtype=adapter.torch_dtype)
    model.eval()
    print(f"conditioner_hidden_dim: {conditioner.hidden_dim}")
    print(f"conditioner_timestep_proj_dim: {conditioner.timestep_proj_dim}")

    _print_header("Load One Manifest Sample")
    dataset = VideoLatentManifestDataset.from_config(cfg)
    if len(dataset) <= 0:
        raise ValueError("Configured video latent manifest is empty.")
    sample = dataset[0]
    z_clean = _batched_sample_z_clean(sample["z_clean"]).to(
        device=adapter.device,
        dtype=adapter.torch_dtype,
    )
    s_src_tensor = torch.full((z_clean.shape[0],), s_src, device=adapter.device)
    s_tgt_tensor = torch.full((z_clean.shape[0],), s_tgt, device=adapter.device)
    flowmap_pair = build_video_flowmap_pair(
        z_clean=z_clean,
        s_src=s_src_tensor,
        s_tgt=s_tgt_tensor,
    )
    context = adapter.build_video_context({"context": [sample["context"]]})

    print(f"sample_id: {sample.get('meta', {}).get('sample_id')}")
    print(f"z_clean: {_tensor_summary(z_clean)}")
    print(f"z_src: {_tensor_summary(flowmap_pair.z_src)}")
    print(f"target_u: {_tensor_summary(flowmap_pair.target_u)}")
    print(f"s_src: {s_src_tensor.detach().cpu().tolist()}")
    print(f"s_tgt: {s_tgt_tensor.detach().cpu().tolist()}")

    _print_header("LingBot Video Input Contract")
    input_dict = adapter.build_lingbot_video_input(
        model,
        flowmap_pair.z_src,
        flowmap_pair.s_src,
        flowmap_pair.s_tgt,
        context,
    )
    for key, value in input_dict.items():
        print(f"{key}: {_tensor_summary(value)}")

    _print_header("No-Grad Video Forward")
    with torch.no_grad():
        pred_u = adapter.forward_video_flowmap(
            model,
            flowmap_pair.z_src,
            flowmap_pair.s_src,
            flowmap_pair.s_tgt,
            context,
        )
    print(f"pred_u: {_tensor_summary(pred_u)}")
    print(f"pred_u_matches_z_src_shape: {tuple(pred_u.shape) == tuple(flowmap_pair.z_src.shape)}")
    print(f"pred_u_all_finite: {bool(torch.isfinite(pred_u).all().item())}")


def _describe_model(model: Any) -> None:
    print(f"model_class: {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"model_forward_signature: {_safe_signature(getattr(model, 'forward', None))}")
    print(f"patch_size: {getattr(model, 'patch_size', None)}")
    print(f"_time_embed_signature: {_safe_signature(getattr(model, '_time_embed', None))}")

    condition_embedder = getattr(model, "condition_embedder", None)
    time_proj = getattr(condition_embedder, "time_proj", None)
    print(f"condition_embedder: {condition_embedder.__class__.__name__ if condition_embedder else None}")
    print(f"time_proj: {time_proj}")
    print(f"time_proj_out_features: {getattr(time_proj, 'out_features', None)}")

    first_param = next(model.parameters(), None)
    if first_param is not None:
        print(f"first_param_device: {first_param.device}")
        print(f"first_param_dtype: {first_param.dtype}")
    else:
        print("first_param_device: None")
        print("first_param_dtype: None")


def _batched_sample_z_clean(z_clean: Any) -> Any:
    import torch

    if not isinstance(z_clean, torch.Tensor):
        raise TypeError("z_clean must be a torch.Tensor.")
    batched = z_clean.unsqueeze(0)
    if batched.ndim == 5:
        return batched
    if batched.ndim == 6 and batched.shape[1] == 1:
        return batched.squeeze(1)
    raise ValueError(
        "A single manifest z_clean sample must become [B, C, F, H, W] after batching."
    )


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


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required source file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def _format_function_signature(source: str, function_name: str) -> str:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return _signature_from_args(node.args)
    return "<not found>"


def _format_method_signature(source: str, class_name: str, method_name: str) -> str:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return _signature_from_args(child.args)
    return "<not found>"


def _signature_from_args(args: ast.arguments) -> str:
    names = [arg.arg for arg in args.posonlyargs + args.args]
    if args.vararg is not None:
        names.append(f"*{args.vararg.arg}")
    names.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg is not None:
        names.append(f"**{args.kwarg.arg}")
    return f"({', '.join(names)})"


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
