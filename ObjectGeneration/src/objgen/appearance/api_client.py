"""Minimal Chat Completions adapter; never persists credentials or raw errors."""
import base64
import json
import mimetypes
import os
from pathlib import Path
import urllib.error
import urllib.request


class APIError(RuntimeError):
    def __init__(self, status, category):
        self.status = status
        self.category = category
        super().__init__(f"API request failed: status={status}, category={category}")


def credentials(env_file):
    key = os.environ.get("VIBEG_API_KEY")
    if not key:
        for line in Path(env_file).read_text().splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name == "VIBEG_API_KEY":
                key = value.strip().strip("\"'")
                break
    if not key or any(c.isspace() for c in key):
        raise ValueError("VIBEG_API_KEY missing or invalid")
    return key


def image_content(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return {"type": "image_url", "image_url": {
        "url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()}}


class APIClient:
    def __init__(self, config):
        self.config = config
        self.key = credentials(config["api_env_file"])
        self.records = []

    def chat(self, messages, **options):
        payload = {"model": self.config["model"], "messages": messages,
                   "max_completion_tokens": self.config["max_completion_tokens"], **options}
        request = urllib.request.Request(
            self.config["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.config["request_timeout_seconds"]) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            category = {401: "authentication", 403: "forbidden", 402: "billing",
                        429: "rate_limit"}.get(exc.code, "http_error")
            raise APIError(exc.code, category) from None
        except (OSError, ValueError):
            raise APIError(None, "transport_or_invalid_json") from None
        if body.get("error") or not body.get("choices"):
            raise APIError(200, "missing_choices_or_error")
        choice = body["choices"][0]
        self.records.append({"model": body.get("model"), "usage": body.get("usage"),
                             "finish_reason": choice.get("finish_reason")})
        if choice.get("finish_reason") not in ("stop", "tool_calls"):
            raise APIError(200, "incomplete_response")
        return choice["message"]
