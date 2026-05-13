from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger("ComfyUI-Sulphur2")


class Sulphur2PromptEnhancer:
    CATEGORY = "Sulphur2/conditioning"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "enhance"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "enhancer_url": ("STRING", {"default": "http://enhancer:8080/v1/chat/completions"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "timeout_seconds": ("INT", {"default": 180, "min": 5, "max": 600}),
            }
        }

    def enhance(self, prompt: str, enhancer_url: str, enabled: bool, timeout_seconds: int):
        if not enabled or not prompt.strip():
            return (prompt,)

        payload = {
            "model": "sulphur-enhancer",
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": 1024,
            "stream": False,
        }
        request = urllib.request.Request(
            enhancer_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:
                data = json.loads(response.read().decode("utf-8"))
            message = data["choices"][0]["message"]
            enhanced = (message.get("content") or "").strip().strip('"').strip("'")
            reasoning = (message.get("reasoning_content") or "").strip()
            if reasoning:
                log.warning("Prompt enhancer returned reasoning content despite thinking-off settings.")
            return (enhanced or prompt,)
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            log.warning("Prompt enhancer failed; using raw prompt: %s", exc)
            return (prompt,)
