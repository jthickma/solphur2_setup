from __future__ import annotations

import inspect
import logging
from typing import Any

log = logging.getLogger("ComfyUI-Sulphur2")


class MissingNodeError(RuntimeError):
    pass


def snap_dim_32(target: int) -> int:
    return max(32, (int(target) // 32) * 32)


def snap_frames(target_frames: int) -> int:
    if target_frames < 1:
        return 1
    n_nearest = (int(target_frames) - 1 + 4) // 8
    return n_nearest * 8 + 1


def stage1_dim(final_dim: int) -> int:
    return max(32, (int(final_dim) // 2 // 32) * 32)


def merge_model_list(kind: str, defaults: list[str]) -> list[str]:
    names: list[str] = []
    try:
        import folder_paths  # type: ignore

        names = list(folder_paths.get_filename_list(kind))
    except Exception:
        names = []
    merged = list(dict.fromkeys([*defaults, *names]))
    return merged or defaults


def get_node_class(class_type: str) -> type:
    try:
        import nodes  # type: ignore
    except Exception as exc:  # pragma: no cover - only available inside ComfyUI
        raise MissingNodeError(
            "ComfyUI node registry is unavailable. This node must run inside ComfyUI."
        ) from exc

    cls = nodes.NODE_CLASS_MAPPINGS.get(class_type)
    if cls is None:
        raise MissingNodeError(
            f"Required node '{class_type}' is not installed. Install the pinned "
            "ComfyUI-LTXVideo, ComfyUI-KJNodes, and ComfyMath dependencies."
        )
    return cls


def node_callable(instance: object) -> Any:
    function_name = getattr(instance, "FUNCTION", None)
    if not function_name:
        raise MissingNodeError(f"{instance.__class__.__name__} does not expose FUNCTION.")
    return getattr(instance, function_name)


def invoke_node(class_type: str, **inputs: Any) -> tuple[Any, ...]:
    cls = get_node_class(class_type)
    instance = cls()
    fn = node_callable(instance)

    signature = inspect.signature(fn)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_kwargs:
        call_inputs = inputs
    else:
        call_inputs = {
            key: value
            for key, value in inputs.items()
            if key in signature.parameters
        }

    result = fn(**call_inputs)
    if isinstance(result, dict):
        if "result" in result:
            result = result["result"]
        else:
            result = ()
    if not isinstance(result, tuple):
        result = (result,)
    return result


def warn_ood_sulphur(width: int, height: int, frames: int, fps: float, mode: str) -> None:
    duration = frames / max(float(fps), 1.0)
    if width > 1280 or height > 704 or duration > 10.2:
        log.warning(
            "Sulphur-2 request is outside the validated default envelope: "
            "%sx%s, %s frames @ %.2f fps, mode=%s. 1080p x 20s quality mode "
            "is a known audio-mux risk.",
            width,
            height,
            frames,
            fps,
            mode,
        )
