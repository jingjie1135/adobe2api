import json
import base64
import threading
import time
import uuid
import random
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Dict

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
DATA_FILE = CONFIG_DIR / "tokens.json"
LEGACY_DATA_FILE = DATA_DIR / "tokens.json"

TokenRecord = Dict[str, Any]


class TokenManager:
    ERROR_COOLDOWN_SECONDS = 180

    def __init__(self):
        self._lock = threading.Lock()
        self.tokens: List[TokenRecord] = []
        self._rr_index = 0
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        with self._lock:
            source = DATA_FILE if DATA_FILE.exists() else LEGACY_DATA_FILE
            if source.exists():
                try:
                    self.tokens = json.loads(source.read_text(encoding="utf-8"))
                    now_ts = time.time()
                    for t in self.tokens:
                        if not isinstance(t, dict):
                            continue
                        t.setdefault("id", uuid.uuid4().hex[:8])
                        t.setdefault("value", "")
                        t.setdefault("status", "active")
                        t.setdefault("fails", 0)
                        t.setdefault("added_at", now_ts)
                        t.setdefault("error_until", 0)
                    if source == LEGACY_DATA_FILE and not DATA_FILE.exists():
                        DATA_FILE.write_text(
                            json.dumps(self.tokens, indent=2), encoding="utf-8"
                        )
                except Exception:
                    self.tokens = []

    def save(self):
        DATA_FILE.write_text(json.dumps(self.tokens, indent=2), encoding="utf-8")

    def add(self, value: str, meta: Optional[TokenRecord] = None):
        with self._lock:
            value = value.strip()
            if value.startswith("Bearer "):
                value = value[7:].strip()
            meta = dict(meta or {})
            account_id = self.account_id_from_token(value)
            if account_id and not meta.get("account_id"):
                meta["account_id"] = account_id

            for t in self.tokens:
                if t["value"] == value:
                    if meta:
                        t.update(meta)
                        self.save()
                    return t

            new_token = {
                "id": uuid.uuid4().hex[:8],
                "value": value,
                "status": "active",
                "fails": 0,
                "added_at": time.time(),
                "error_until": 0,
            }
            if meta:
                new_token.update(meta)
            self.tokens.append(new_token)
            self.save()
            return new_token

    def upsert_auto_refresh_token(
        self,
        value: str,
        profile_id: str,
        profile_name: Optional[str] = None,
        profile_email: Optional[str] = None,
    ):
        with self._lock:
            value = value.strip()
            if value.startswith("Bearer "):
                value = value[7:].strip()

            now_ts = time.time()
            pid = str(profile_id or "").strip()
            account_id = self.account_id_from_token(value)
            if not pid:
                raise ValueError("profile_id is required")

            target = None
            for t in self.tokens:
                if (
                    t.get("auto_refresh") is True
                    and str(t.get("refresh_profile_id") or "").strip() == pid
                ):
                    target = t
                    break

            if target is not None:
                was_disabled = target.get("status") == "disabled"
                target["value"] = value
                target["status"] = "disabled" if was_disabled else "active"
                target["fails"] = 0
                target["error_until"] = 0
                target["updated_at"] = now_ts
                target["source"] = "auto_refresh"
                target["auto_refresh"] = True
                target["refresh_profile_id"] = pid
                target["refresh_profile_name"] = str(profile_name or "").strip() or pid
                target["refresh_profile_email"] = str(profile_email or "").strip()
                if account_id:
                    target["account_id"] = account_id
                self.save()
                return dict(target)

            new_token = {
                "id": uuid.uuid4().hex[:8],
                "value": value,
                "status": "active",
                "fails": 0,
                "added_at": now_ts,
                "updated_at": now_ts,
                "error_until": 0,
                "source": "auto_refresh",
                "auto_refresh": True,
                "refresh_profile_id": pid,
                "refresh_profile_name": str(profile_name or "").strip() or pid,
                "refresh_profile_email": str(profile_email or "").strip(),
                "account_id": account_id,
            }
            self.tokens.append(new_token)
            self.save()
            return dict(new_token)

    def remove(self, tid: str):
        with self._lock:
            self.tokens = [t for t in self.tokens if t["id"] != tid]
            self.save()

    def remove_auto_refresh_by_profile(self, profile_id: str):
        pid = str(profile_id or "").strip()
        if not pid:
            return
        with self._lock:
            self.tokens = [
                t
                for t in self.tokens
                if not (
                    t.get("auto_refresh") is True
                    and str(t.get("refresh_profile_id") or "").strip() == pid
                )
            ]
            self.save()

    def get_by_id(self, tid: str) -> Optional[TokenRecord]:
        with self._lock:
            for t in self.tokens:
                if t.get("id") == tid:
                    return dict(t)
        return None

    def get_meta_by_value(self, value: str) -> TokenRecord:
        token_value = str(value or "").strip()
        with self._lock:
            for t in self.tokens:
                if str(t.get("value") or "").strip() != token_value:
                    continue
                return {
                    "token_id": t.get("id"),
                    "token_account_id": t.get("account_id") or self.account_id_from_token(token_value),
                    "token_account_name": t.get("refresh_profile_name") or "",
                    "token_account_email": t.get("refresh_profile_email") or "",
                    "token_source": t.get("source") or "manual",
                }
        return {
            "token_id": "",
            "token_account_id": "",
            "token_account_name": "",
            "token_account_email": "",
            "token_source": "manual",
        }

    def set_status(self, tid: str, status: str):
        with self._lock:
            for t in self.tokens:
                if t["id"] == tid:
                    t["status"] = status
                    t["fails"] = 0 if status == "active" else t["fails"]
                    if status == "active":
                        t["error_until"] = 0
            self.save()

    def set_credits(self, tid: str, credits: TokenRecord):
        with self._lock:
            for t in self.tokens:
                if t.get("id") != tid:
                    continue
                t["credits_total"] = credits.get("total")
                t["credits_used"] = credits.get("used")
                t["credits_available"] = credits.get("available")
                t["credits_available_until"] = credits.get("available_until")
                t["credits_updated_at"] = credits.get("updated_at") or int(time.time())
                t["credits_error"] = ""
                self.save()
                return dict(t)
        return None

    def set_credits_error(self, tid: str, error_message: str):
        with self._lock:
            for t in self.tokens:
                if t.get("id") != tid:
                    continue
                t["credits_error"] = str(error_message or "")[:300]
                t["credits_updated_at"] = int(time.time())
                self.save()
                return dict(t)
        return None

    def list_active_ids(self) -> List[str]:
        with self._lock:
            return [
                str(t.get("id") or "")
                for t in self.tokens
                if t.get("status") == "active"
            ]

    def _pick_active_token_locked(
        self, strategy: str = "round_robin"
    ) -> Optional[TokenRecord]:
        active = [t for t in self.tokens if t.get("status") in {"active", "error"}]
        if not active:
            return None

        chosen = None
        mode = str(strategy or "round_robin").strip().lower()
        if mode == "random":
            chosen = random.choice(active)
        else:
            idx = self._rr_index % len(active)
            chosen = active[idx]
            self._rr_index = (idx + 1) % len(active)
        return chosen

    def get_available(self, strategy: str = "round_robin") -> Optional[str]:
        with self._lock:
            chosen = self._pick_active_token_locked(strategy=strategy)
            return chosen["value"] if chosen is not None else None

    @classmethod
    def account_id_from_token(cls, value: str) -> str:
        data = cls._decode_jwt_payload(value)
        if not data:
            return ""
        return str(
            data.get("user_id") or data.get("aa_id") or data.get("sub") or ""
        ).strip()

    def get_available_for_account(
        self, account_id: str, strategy: str = "round_robin"
    ) -> Optional[str]:
        aid = str(account_id or "").strip()
        if not aid:
            return None
        with self._lock:
            active = [
                t
                for t in self.tokens
                if t.get("status") in {"active", "error"}
                and str(t.get("account_id") or self.account_id_from_token(t.get("value") or ""))
                == aid
            ]
            if not active:
                return None
            mode = str(strategy or "round_robin").strip().lower()
            if mode == "random":
                return random.choice(active)["value"]
            idx = self._rr_index % len(active)
            self._rr_index = (self._rr_index + 1) % max(1, len(self.tokens))
            return active[idx]["value"]

    def list_active_account_tokens(self) -> List[TokenRecord]:
        with self._lock:
            items = []
            seen = set()
            for t in self.tokens:
                if t.get("status") != "active":
                    continue
                value = str(t.get("value") or "").strip()
                aid = str(t.get("account_id") or self.account_id_from_token(value)).strip()
                if not value or not aid or aid in seen:
                    continue
                seen.add(aid)
                items.append(
                    {
                        "token": value,
                        "account_id": aid,
                        "account_name": str(t.get("refresh_profile_name") or ""),
                        "account_email": str(t.get("refresh_profile_email") or ""),
                    }
                )
            return items

    def report_exhausted(self, value: str):
        with self._lock:
            for t in self.tokens:
                if t["value"] == value:
                    t["status"] = "exhausted"
                    t["error_until"] = 0
            self.save()

    def report_invalid(self, value: str):
        with self._lock:
            for t in self.tokens:
                if t["value"] == value:
                    t["status"] = "invalid"
                    t["error_until"] = 0
            self.save()

    def handle_auth_failure(self, value: str) -> TokenRecord:
        token_value = str(value or "").strip()
        linked_profile_id = ""
        linked_auto_refresh = False

        with self._lock:
            for t in self.tokens:
                if str(t.get("value") or "").strip() != token_value:
                    continue
                linked_profile_id = str(t.get("refresh_profile_id") or "").strip()
                linked_auto_refresh = bool(t.get("auto_refresh"))
                break

        if not linked_auto_refresh or not linked_profile_id:
            self.report_invalid(token_value)
            return {
                "status": "invalid",
                "message": "token invalid or expired",
                "http_status": 401,
                "profile_id": linked_profile_id,
            }

        try:
            from core.refresh_mgr import refresh_manager

            if refresh_manager.is_profile_enabled(linked_profile_id) is False:
                self.report_invalid(token_value)
                return {
                    "status": "invalid",
                    "message": "token expired and auto refresh is disabled",
                    "http_status": 401,
                    "profile_id": linked_profile_id,
                }

            refresh_result = refresh_manager.refresh_once(linked_profile_id)
        except Exception as exc:
            self.report_error(token_value)
            return {
                "status": "retry",
                "message": f"auto refresh failed: {exc}",
                "http_status": None,
                "profile_id": linked_profile_id,
            }

        return {
            "status": "refreshed",
            "message": "token refreshed via cookie",
            "http_status": 200,
            "profile_id": linked_profile_id,
            "result": refresh_result,
        }

    def report_error(self, value: str):
        with self._lock:
            for t in self.tokens:
                if t["value"] == value:
                    t["fails"] += 1
                    t["updated_at"] = time.time()
            self.save()

    def report_success(self, value: str):
        with self._lock:
            for t in self.tokens:
                if t["value"] == value:
                    t["fails"] = 0
                    if t["status"] == "error":
                        t["status"] = "active"
                        t["error_until"] = 0
            self.save()

    @staticmethod
    def _decode_jwt_payload(value: str) -> Optional[TokenRecord]:
        token = str(value or "").strip()
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1]
        payload += "=" * ((4 - len(payload) % 4) % 4)
        try:
            raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
            data = json.loads(raw.decode("utf-8", errors="ignore"))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    @classmethod
    def _decode_jwt_exp(cls, value: str) -> Optional[int]:
        data = cls._decode_jwt_payload(value)
        if not data:
            return None

        exp = data.get("exp")
        if isinstance(exp, (int, float)):
            return int(exp)

        # Adobe tokens often expose created_at + expires_in in payload instead of exp.
        created_at = data.get("created_at")
        expires_in = data.get("expires_in")
        try:
            created_at_val = int(str(created_at).strip())
            expires_in_val = int(str(expires_in).strip())
        except Exception:
            return None

        if created_at_val <= 0 or expires_in_val <= 0:
            return None

        # Some fields are milliseconds (e.g. 1771862511913 / 86400000)
        if created_at_val > 10_000_000_000:
            created_at_val = int(created_at_val / 1000)
        if expires_in_val > 86400 * 2:
            expires_in_val = int(expires_in_val / 1000)

        return created_at_val + expires_in_val

    def list_all(self):
        with self._lock:
            res = []
            now_ts = int(time.time())
            for t in self.tokens:
                # mask value
                val = t["value"]
                masked = val[:15] + "..." + val[-10:] if len(val) > 30 else "***"
                exp_ts = self._decode_jwt_exp(val)
                remaining_seconds = None
                exp_readable = None
                if exp_ts is not None:
                    remaining_seconds = exp_ts - now_ts
                    try:
                        exp_readable = datetime.fromtimestamp(exp_ts).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    except Exception:
                        exp_readable = str(exp_ts)
                res.append(
                    {
                        "id": t["id"],
                        "value": masked,
                        "status": t["status"],
                        "fails": t["fails"],
                        "added_at": t["added_at"],
                        "error_until": t.get("error_until", 0),
                        "source": t.get("source", "manual"),
                        "auto_refresh": bool(t.get("auto_refresh", False)),
                        "refresh_profile_id": t.get("refresh_profile_id"),
                        "refresh_profile_name": t.get("refresh_profile_name"),
                        "refresh_profile_email": t.get("refresh_profile_email"),
                        "credits_total": t.get("credits_total"),
                        "credits_used": t.get("credits_used"),
                        "credits_available": t.get("credits_available"),
                        "credits_available_until": t.get("credits_available_until"),
                        "credits_updated_at": t.get("credits_updated_at"),
                        "credits_error": t.get("credits_error", ""),
                        "expires_at": exp_ts,
                        "expires_at_text": exp_readable,
                        "remaining_seconds": remaining_seconds,
                        "is_expired": bool(
                            exp_ts is not None
                            and remaining_seconds is not None
                            and remaining_seconds <= 0
                        ),
                    }
                )
            return res

    def export_tokens(self, ids: Optional[List[str]] = None) -> List[TokenRecord]:
        selected_ids = None
        if isinstance(ids, list):
            normalized = [str(x or "").strip() for x in ids]
            selected_ids = {x for x in normalized if x}
        with self._lock:
            out: List[TokenRecord] = []
            for t in self.tokens:
                tid = str(t.get("id") or "").strip()
                if selected_ids is not None and tid not in selected_ids:
                    continue
                out.append(
                    {
                        "id": tid,
                        "token": str(t.get("value") or "").strip(),
                        "status": str(t.get("status") or "active"),
                        "source": str(t.get("source") or "manual"),
                        "auto_refresh": bool(t.get("auto_refresh", False)),
                        "refresh_profile_id": t.get("refresh_profile_id"),
                        "refresh_profile_name": t.get("refresh_profile_name"),
                        "refresh_profile_email": t.get("refresh_profile_email"),
                        "added_at": t.get("added_at"),
                    }
                )
            return out


token_manager = TokenManager()
