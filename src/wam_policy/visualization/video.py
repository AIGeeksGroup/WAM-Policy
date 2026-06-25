from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping

import torch


@dataclass(frozen=True)
class VideoEncodingResult:
    path: Path
    backend: str
    codec: str
    fps: int
    succeeded: bool
    fallback_level: str
    error: str | None = None

    def to_summary(self) -> dict[str, Any]:
        summary = asdict(self)
        summary["path"] = str(self.path)
        return summary


def to_uint8_frame_array(frames: Any) -> Any:
    """Normalize common tensor/array frame layouts to uint8 [T, H, W, C]."""
    np = _numpy()
    if isinstance(frames, torch.Tensor):
        tensor = frames.detach().cpu()
        if tensor.ndim == 5 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim == 4 and tensor.shape[-1] in {1, 3}:
            pass
        elif tensor.ndim == 4 and tensor.shape[0] in {1, 3}:
            tensor = tensor.permute(1, 2, 3, 0)
        if tensor.ndim != 4:
            raise ValueError("Video tensor must have shape [T,H,W,C], [C,T,H,W], or [1,T,H,W,C].")
        if tensor.dtype != torch.uint8:
            tensor = tensor.float()
            if tensor.numel() > 0 and tensor.min().item() < 0.0:
                tensor = (tensor + 1.0) / 2.0
            tensor = tensor.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        return tensor.numpy()

    if isinstance(frames, np.ndarray):
        array = frames
        if array.ndim == 5 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4 and array.shape[-1] in {1, 3}:
            pass
        elif array.ndim == 4 and array.shape[0] in {1, 3}:
            array = np.moveaxis(array, 0, -1)
        if array.ndim != 4:
            raise ValueError("Video array must have shape [T,H,W,C], [C,T,H,W], or [1,T,H,W,C].")
        if array.dtype != np.uint8:
            array = array.astype("float32", copy=False)
            if array.size and float(array.min()) < 0.0:
                array = (array + 1.0) / 2.0
            array = (array.clip(0.0, 1.0) * 255.0).round().astype("uint8")
        return array

    raise TypeError("frames must be a torch.Tensor or numpy.ndarray.")


def stitch_chunks(chunks: Iterable[Any]) -> Any:
    np = _numpy()
    arrays = [to_uint8_frame_array(chunk) for chunk in chunks]
    if not arrays:
        raise ValueError("At least one chunk is required.")
    return np.concatenate(arrays, axis=0)


def contact_sheet_video(chunks: Iterable[Any], *, columns: int | None = None) -> Any:
    np = _numpy()
    arrays = [to_uint8_frame_array(chunk) for chunk in chunks]
    if not arrays:
        raise ValueError("At least one chunk is required.")
    if columns is None:
        columns = max(1, math.ceil(math.sqrt(len(arrays))))
    if columns <= 0:
        raise ValueError("columns must be positive.")

    rows = math.ceil(len(arrays) / columns)
    max_frames = max(array.shape[0] for array in arrays)
    max_height = max(array.shape[1] for array in arrays)
    max_width = max(array.shape[2] for array in arrays)
    channels = arrays[0].shape[3]
    output = np.zeros(
        (max_frames, rows * max_height, columns * max_width, channels),
        dtype=np.uint8,
    )
    for chunk_index, array in enumerate(arrays):
        row = chunk_index // columns
        col = chunk_index % columns
        y0 = row * max_height
        x0 = col * max_width
        for frame_index in range(max_frames):
            frame = array[min(frame_index, array.shape[0] - 1)]
            output[
                frame_index,
                y0 : y0 + frame.shape[0],
                x0 : x0 + frame.shape[1],
                :,
            ] = frame
    return output


def side_by_side(left: Any, right: Any, *, pad_px: int = 4) -> Any:
    np = _numpy()
    left_array = to_uint8_frame_array(left)
    right_array = to_uint8_frame_array(right)
    frame_count = max(left_array.shape[0], right_array.shape[0])
    height = max(left_array.shape[1], right_array.shape[1])
    left_width = left_array.shape[2]
    right_width = right_array.shape[2]
    channels = left_array.shape[3]
    output = np.zeros(
        (frame_count, height, left_width + pad_px + right_width, channels),
        dtype=np.uint8,
    )
    for frame_index in range(frame_count):
        left_frame = left_array[min(frame_index, left_array.shape[0] - 1)]
        right_frame = right_array[min(frame_index, right_array.shape[0] - 1)]
        output[frame_index, : left_frame.shape[0], :left_width, :] = left_frame
        output[
            frame_index,
            : right_frame.shape[0],
            left_width + pad_px : left_width + pad_px + right_width,
            :,
        ] = right_frame
    return output


def write_mp4(frames: Any, output_path: str | Path, *, fps: int) -> Path:
    return write_video_artifact(frames, output_path, fps=fps).path


def write_video_artifact(frames: Any, output_path: str | Path, *, fps: int) -> VideoEncodingResult:
    if fps <= 0:
        raise ValueError("fps must be positive.")
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    array = to_uint8_frame_array(frames)

    last_error: str | None = None
    for candidate in _ffmpeg_encoding_candidates(path):
        try:
            _write_ffmpeg_rawvideo(array, candidate["path"], fps=fps, candidate=candidate)
            return VideoEncodingResult(
                path=candidate["path"],
                backend=str(candidate["backend"]),
                codec=str(candidate["codec"]),
                fps=fps,
                succeeded=True,
                fallback_level=str(candidate["fallback_level"]),
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    try:
        _write_imageio_mp4(array, path, fps=fps)
        return VideoEncodingResult(
            path=path,
            backend="imageio-ffmpeg",
            codec="libx264_or_default",
            fps=fps,
            succeeded=True,
            fallback_level="cpu",
            error=last_error,
        )
    except Exception as exc:
        last_error = f"{last_error}; imageio-mp4 {type(exc).__name__}: {exc}"

    gif_path = path.with_suffix(".gif")
    try:
        _imageio().mimsave(str(gif_path), list(array), fps=fps)
        return VideoEncodingResult(
            path=gif_path,
            backend="imageio-gif",
            codec="gif",
            fps=fps,
            succeeded=True,
            fallback_level="gif",
            error=last_error,
        )
    except Exception as exc:
        last_error = f"{last_error}; gif {type(exc).__name__}: {exc}"

    frames_dir = path.with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        imageio = _imageio()
        for index, frame in enumerate(array):
            imageio.imwrite(str(frames_dir / f"frame_{index:06d}.png"), frame)
        return VideoEncodingResult(
            path=frames_dir,
            backend="imageio-png-sequence",
            codec="png",
            fps=fps,
            succeeded=True,
            fallback_level="png_sequence",
            error=last_error,
        )
    except Exception as exc:
        last_error = f"{last_error}; png_sequence {type(exc).__name__}: {exc}"

    tensor_path = path.with_suffix(".pt")
    torch.save(torch.from_numpy(array), tensor_path)
    return VideoEncodingResult(
        path=tensor_path,
        backend="torch-save",
        codec="uint8_tensor",
        fps=fps,
        succeeded=True,
        fallback_level="tensor",
        error=last_error,
    )


def probe_video_encoder(
    output_dir: str | Path,
    *,
    fps: int = 10,
) -> VideoEncodingResult:
    np = _numpy()
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    frames[1, :, :, 0] = 255
    return write_video_artifact(frames, Path(output_dir) / "video_encoder_probe.mp4", fps=fps)


def write_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeError(
            "Video visualization requires numpy. Install visualization dependencies on the server."
        ) from exc


def _imageio() -> Any:
    try:
        return importlib.import_module("imageio.v2")
    except ImportError as exc:
        raise RuntimeError(
            "Writing MP4 visualization requires imageio with ffmpeg support. "
            "Install imageio[ffmpeg] or use the LingBot/RoboTwin runtime environment."
        ) from exc


def _write_imageio_mp4(array: Any, path: Path, *, fps: int) -> None:
    imageio = _imageio()
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1)
    try:
        for frame in array:
            writer.append_data(frame)
    finally:
        writer.close()


def _ffmpeg_encoding_candidates(path: Path) -> list[dict[str, Any]]:
    ffmpeg = shutil.which(os.environ.get("WAM_FFMPEG_BIN", "ffmpeg"))
    if ffmpeg is None:
        return []

    encoders = _ffmpeg_encoders(ffmpeg)
    candidates: list[dict[str, Any]] = []
    vaapi_device = os.environ.get("WAM_VAAPI_DEVICE", "/dev/dri/renderD128")
    if "h264_vaapi" in encoders and Path(vaapi_device).exists():
        candidates.append(
            {
                "backend": "ffmpeg-vaapi",
                "codec": "h264_vaapi",
                "fallback_level": "gpu",
                "path": path,
                "cmd_extra": [
                    "-vaapi_device",
                    vaapi_device,
                    "-vf",
                    "format=nv12,hwupload",
                    "-c:v",
                    "h264_vaapi",
                    "-qp",
                    "23",
                ],
            }
        )
    if "h264_amf" in encoders:
        candidates.append(
            {
                "backend": "ffmpeg-amf",
                "codec": "h264_amf",
                "fallback_level": "gpu",
                "path": path,
                "cmd_extra": ["-c:v", "h264_amf"],
            }
        )
    if "libx264" in encoders:
        candidates.append(
            {
                "backend": "ffmpeg-cpu",
                "codec": "libx264",
                "fallback_level": "cpu",
                "path": path,
                "cmd_extra": ["-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23"],
            }
        )
    elif "mpeg4" in encoders:
        candidates.append(
            {
                "backend": "ffmpeg-cpu",
                "codec": "mpeg4",
                "fallback_level": "cpu",
                "path": path,
                "cmd_extra": ["-vcodec", "mpeg4"],
            }
        )
    return candidates


def _ffmpeg_encoders(ffmpeg: str) -> set[str]:
    try:
        completed = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return set()
    text = f"{completed.stdout}\n{completed.stderr}"
    encoders: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def _write_ffmpeg_rawvideo(
    array: Any,
    path: Path,
    *,
    fps: int,
    candidate: Mapping[str, Any],
) -> None:
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("ffmpeg rawvideo writer requires [T,H,W,3] uint8 frames.")
    ffmpeg = shutil.which(os.environ.get("WAM_FFMPEG_BIN", "ffmpeg"))
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable not found.")
    height, width = int(array.shape[1]), int(array.shape[2])
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        *list(candidate["cmd_extra"]),
        str(path),
    ]
    completed = subprocess.run(
        cmd,
        input=array.tobytes(),
        check=False,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"{candidate['backend']} failed with exit {completed.returncode}: {stderr}"
        )
