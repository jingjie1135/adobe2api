from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

import core.refresh_mgr as refresh_mgr_module
import core.token_mgr as token_mgr_module


class FakeResponse:
    status_code: int
    _payload: dict[str, object]
    text: str

    def __init__(
        self, status_code: int, payload: dict[str, object], text: str = ""
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload)

    def json(self) -> dict[str, object]:
        return self._payload


class RefreshHarness:
    tmp: tempfile.TemporaryDirectory[str]
    config_dir: Path
    old_token_data_file: Path
    old_token_legacy_file: Path
    old_profile_file: Path
    old_refresh_token_manager: token_mgr_module.TokenManager
    old_refresh_manager: refresh_mgr_module.RefreshManager
    old_post: object
    old_get: object
    token_manager: token_mgr_module.TokenManager
    refresh_manager: refresh_mgr_module.RefreshManager
    queued_tokens: list[str]
    post_calls: int

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.tmp.name)

        self.old_token_data_file = token_mgr_module.DATA_FILE
        self.old_token_legacy_file = token_mgr_module.LEGACY_DATA_FILE
        self.old_profile_file = refresh_mgr_module.PROFILE_FILE
        self.old_refresh_token_manager = refresh_mgr_module.token_manager
        self.old_refresh_manager = refresh_mgr_module.refresh_manager
        self.old_post = refresh_mgr_module.requests.post
        self.old_get = refresh_mgr_module.requests.get
        self.token_manager = cast(token_mgr_module.TokenManager, object())
        self.refresh_manager = cast(refresh_mgr_module.RefreshManager, object())
        self.queued_tokens = []
        self.post_calls = 0

    def __enter__(self) -> RefreshHarness:
        token_mgr_module.DATA_FILE = self.config_dir / "tokens.json"
        token_mgr_module.LEGACY_DATA_FILE = self.config_dir / "legacy_tokens.json"
        refresh_mgr_module.PROFILE_FILE = self.config_dir / "refresh_profile.json"

        self.token_manager = token_mgr_module.TokenManager()
        self.refresh_manager = refresh_mgr_module.RefreshManager()
        refresh_mgr_module.token_manager = self.token_manager
        refresh_mgr_module.refresh_manager = self.refresh_manager

        self.queued_tokens = []
        self.post_calls = 0
        refresh_mgr_module.requests.post = self.fake_post
        refresh_mgr_module.requests.get = self.fake_get
        return self

    def __exit__(self, *_args: object) -> None:
        refresh_mgr_module.requests.post = self.old_post
        refresh_mgr_module.requests.get = self.old_get
        refresh_mgr_module.refresh_manager = self.old_refresh_manager
        refresh_mgr_module.token_manager = self.old_refresh_token_manager
        refresh_mgr_module.PROFILE_FILE = self.old_profile_file
        token_mgr_module.LEGACY_DATA_FILE = self.old_token_legacy_file
        token_mgr_module.DATA_FILE = self.old_token_data_file
        self.tmp.cleanup()

    def fake_post(self, *_args: object, **_kwargs: object) -> FakeResponse:
        self.post_calls += 1
        if not self.queued_tokens:
            raise AssertionError("no queued token for refresh")
        return FakeResponse(
            200,
            {"access_token": self.queued_tokens.pop(0), "expires_in": 86_400},
        )

    def fake_get(self, *_args: object, **kwargs: object) -> FakeResponse:
        headers_obj = kwargs.get("headers")
        headers = headers_obj if isinstance(headers_obj, dict) else {}
        token = str(headers.get("Authorization") or "").replace("Bearer ", "").strip()
        user_id = self.token_manager.account_id_from_token(token)
        return FakeResponse(
            200,
            {
                "displayName": f"Account {user_id}",
                "email": f"{user_id}@example.test",
                "userId": user_id,
            },
        )


def make_token(user_id: str, marker: str) -> str:
    def encode(data: dict[str, object]) -> str:
        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return ".".join(
        [
            encode({"alg": "none"}),
            encode({"user_id": user_id, "exp": 4_102_444_800, "marker": marker}),
            "signature",
        ]
    )


class RefreshTokenLifecycleTests(unittest.TestCase):
    def test_manual_refresh_works_when_auto_refresh_is_disabled_and_preserves_disabled_token_status(self) -> None:
        with RefreshHarness() as harness:
            profile = cast(
                dict[str, object],
                harness.refresh_manager.import_cookie("sid=one", name="Account one"),
            )
            profile_id = str(profile["id"])
            harness.queued_tokens.append(make_token("user-1", "first"))

            harness.refresh_manager.refresh_once(
                profile_id, allow_disabled_profile=True
            )
            token_id = str(harness.token_manager.tokens[0]["id"])
            harness.token_manager.set_status(token_id, "disabled")
            harness.refresh_manager.set_enabled(profile_id, False)
            second_token = make_token("user-1", "second")
            harness.queued_tokens.append(second_token)

            harness.refresh_manager.refresh_once(
                profile_id, allow_disabled_profile=True
            )

            refreshed = harness.token_manager.get_by_id(token_id)
            self.assertIsNotNone(refreshed)
            assert refreshed is not None
            self.assertEqual("disabled", refreshed["status"])
            self.assertEqual(second_token, refreshed["value"])

    def test_disabled_auto_refresh_prevents_auth_failure_recovery(self) -> None:
        with RefreshHarness() as harness:
            profile = cast(
                dict[str, object],
                harness.refresh_manager.import_cookie("sid=one", name="Account one"),
            )
            profile_id = str(profile["id"])
            first_token = make_token("user-1", "first")
            harness.queued_tokens.append(first_token)
            harness.refresh_manager.refresh_once(
                profile_id, allow_disabled_profile=True
            )
            harness.refresh_manager.set_enabled(profile_id, False)
            harness.post_calls = 0

            result = harness.token_manager.handle_auth_failure(first_token)

            self.assertEqual("invalid", result["status"])
            self.assertIn("disabled", result["message"])
            self.assertEqual(0, harness.post_calls)

    def test_reimported_cookie_for_same_account_updates_existing_profile_without_enabling_auto_refresh(self) -> None:
        with RefreshHarness() as harness:
            profile = cast(
                dict[str, object],
                harness.refresh_manager.import_cookie("sid=old", name="Original"),
            )
            profile_id = str(profile["id"])
            harness.queued_tokens.append(make_token("user-1", "old"))
            harness.refresh_manager.refresh_once(
                profile_id, allow_disabled_profile=True
            )
            harness.refresh_manager.set_enabled(profile_id, False)

            imported = cast(
                dict[str, object],
                harness.refresh_manager.import_cookie("sid=new", name="Updated"),
            )
            new_token = make_token("user-1", "new")
            harness.queued_tokens.append(new_token)
            result = harness.refresh_manager.refresh_once(
                str(imported["id"]), allow_disabled_profile=True
            )

            profiles = harness.refresh_manager.list_profiles()
            self.assertEqual(1, len(profiles))
            self.assertEqual(profile_id, profiles[0]["id"])
            self.assertFalse(profiles[0]["enabled"])
            self.assertEqual(profile_id, result["profile_id"])
            self.assertEqual(1, len(harness.token_manager.tokens))
            self.assertEqual(
                profile_id, harness.token_manager.tokens[0]["refresh_profile_id"]
            )
            self.assertEqual(new_token, harness.token_manager.tokens[0]["value"])

    def test_same_account_merge_removes_token_bound_to_removed_profile(self) -> None:
        with RefreshHarness() as harness:
            canonical_profile = cast(
                dict[str, object],
                harness.refresh_manager.import_cookie("sid=canonical", name="Canonical"),
            )
            canonical_profile_id = str(canonical_profile["id"])
            harness.queued_tokens.append(make_token("user-1", "canonical"))
            harness.refresh_manager.refresh_once(
                canonical_profile_id, allow_disabled_profile=True
            )

            removed_profile = cast(
                dict[str, object],
                harness.refresh_manager.import_cookie("sid=removed", name="Removed"),
            )
            removed_profile_id = str(removed_profile["id"])
            harness.token_manager.upsert_auto_refresh_token(
                make_token("user-1", "stale"),
                profile_id=removed_profile_id,
                profile_name="Removed",
                profile_email="user-1@example.test",
            )

            replacement_token = make_token("user-1", "replacement")
            harness.queued_tokens.append(replacement_token)
            result = harness.refresh_manager.refresh_once(
                removed_profile_id, allow_disabled_profile=True
            )

            self.assertEqual(canonical_profile_id, result["profile_id"])
            self.assertEqual(1, len(harness.refresh_manager.list_profiles()))
            self.assertFalse(
                any(
                    token.get("refresh_profile_id") == removed_profile_id
                    for token in harness.token_manager.tokens
                )
            )
            self.assertEqual(1, len(harness.token_manager.tokens))
            self.assertEqual(
                canonical_profile_id,
                harness.token_manager.tokens[0]["refresh_profile_id"],
            )
            self.assertEqual(replacement_token, harness.token_manager.tokens[0]["value"])


if __name__ == "__main__":
    _ = unittest.main()
