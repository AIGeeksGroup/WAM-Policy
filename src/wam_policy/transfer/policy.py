"""Stage 1.1 trainability: the full transformer is trained, except the unused serialization-only Conv3d."""

from collections import defaultdict


def parameter_group(name):
    if name.startswith("patch_embedding."):
        return "checkpoint_only_conv3d"
    for prefix, group in [
        ("patch_embedding_mlp.", "video_input"),
        ("proj_out.", "video_output"),
        ("action_embedder.", "action_input"),
        ("action_proj_out.", "action_output"),
        ("condition_embedder_action.", "action_conditioning"),
        ("condition_embedder.", "video_conditioning"),
        ("blocks.", "shared_transformer"),
    ]:
        if name.startswith(prefix):
            return group
    return "shared_output_conditioning"


def apply_trainability(model, *, vae=None, text_encoder=None):
    groups = defaultdict(
        lambda: {"parameters": 0, "trainable_parameters": 0, "names": []}
    )
    for name, p in model.named_parameters():
        group = parameter_group(name)
        p.requires_grad_(group != "checkpoint_only_conv3d")
        groups[group]["parameters"] += p.numel()
        groups[group]["trainable_parameters"] += p.numel() if p.requires_grad else 0
        groups[group]["names"].append(name)
    for component in (vae, text_encoder):
        if component is not None:
            component.eval().requires_grad_(False)
    return dict(groups)
