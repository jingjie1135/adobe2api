import base64
import json
import unittest
import uuid
from collections.abc import Mapping
from typing import cast

from core.adobe_client import AdobeClient, format_upstream_status_message


class ArpSessionHeaderTests(unittest.TestCase):
    def _decode_arp_header(self, value: str) -> dict[str, object]:
        decoded = base64.b64decode(value).decode("utf-8")
        raw_data = cast(object, json.loads(decoded))
        self.assertIsInstance(raw_data, dict)
        return cast(dict[str, object], raw_data)

    def assert_valid_arp_header(self, headers: Mapping[str, str]) -> None:
        value = headers.get("x-arp-session-id", "")
        self.assertIsInstance(value, str)
        self.assertTrue(value.strip())

        data = self._decode_arp_header(value)
        _ = uuid.UUID(str(data.get("sid") or ""))
        self.assertRegex(
            str(data.get("ftr") or ""),
            r"^[0-9a-f]{32}_[0-9]{13}_[0-9]+_dUAL43-mnts-ants-d4_31ck__tt$",
        )

    def test_image_and_video_submit_headers_include_valid_arp_session_id(self):
        client = AdobeClient()

        self.assert_valid_arp_header(client._submit_headers("token", prompt="hello"))
        self.assert_valid_arp_header(client._video_submit_headers("token"))

    def test_408_status_message_mentions_arp_session_header(self):
        message = format_upstream_status_message("submit failed", 408, "timeout")

        self.assertIn("408", message)
        self.assertIn("x-arp-session-id", message)

if __name__ == "__main__":
    _ = unittest.main()
