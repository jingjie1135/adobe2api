from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Iterator

from core.assistant_errors import AssistantClientError

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class AssistantImageResult:
    url: str
    creative_cloud_file_id: str
    model_id: str
    model_version: str
    width: int
    height: int
    content_type: str
    request_id: str
    prompt: str


def _iter_sse_data(events_text: str) -> Iterator[dict[str, JsonValue]]:
    for line in str(events_text or "").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line.split(":", 1)[1].strip()
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            yield data


def _image_result_from_payload(payload: dict[str, JsonValue]) -> AssistantImageResult:
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise AssistantClientError("assistant output did not include images")
    image = images[0]
    if not isinstance(image, dict):
        raise AssistantClientError("assistant image output is invalid")
    url = str(image.get("presignedUrl") or "").strip()
    if not url:
        raise AssistantClientError("assistant image output missing presignedUrl")
    return AssistantImageResult(
        url=url,
        creative_cloud_file_id=str(image.get("creativeCloudFileId") or "").strip(),
        model_id=str(image.get("modelId") or "").strip(),
        model_version=str(image.get("modelVersion") or "").strip(),
        width=int(image.get("width") or 0),
        height=int(image.get("height") or 0),
        content_type=str(image.get("contentType") or "").strip(),
        request_id=str(image.get("requestId") or "").strip(),
        prompt=str(image.get("prompt") or "").strip(),
    )


def extract_gpt_image_result_from_events(events_text: str) -> AssistantImageResult:
    for event in _iter_sse_data(events_text):
        if str(event.get("type") or "") != "tool-completed":
            continue
        if str(event.get("toolName") or "") != "generate_image_gpt_image_2":
            continue
        if event.get("success") is not True:
            raise AssistantClientError("assistant gpt-image tool failed")
        output = str(event.get("output") or "").strip()
        if not output:
            raise AssistantClientError("assistant gpt-image tool returned empty output")
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as exc:
            raise AssistantClientError("assistant gpt-image output is invalid json") from exc
        if not isinstance(payload, dict):
            raise AssistantClientError("assistant gpt-image output is invalid")
        return _image_result_from_payload(payload)
    raise AssistantClientError("assistant gpt-image result was not found")
