import json
import hashlib
import importlib
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from core.config_mgr import config_manager


BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
PROFILE_FILE = CONFIG_DIR / "refresh_profile.json"

RefreshRecord = Dict[str, Any]
token_manager: Any = None


def _token_manager() -> Any:
    global token_manager
    if token_manager is None:
        token_manager = getattr(importlib.import_module("core.token_mgr"), "token_manager")
    return token_manager


class RefreshManager:
    DEFAULT_REFRESH_URL = "https://adobeid-na1.services.adobe.com/ims/check/v6/token?jslVersion=v2-v0.48.0-1-g1e322cb"
    DEFAULT_SCOPE = (
        "AdobeID,firefly_api,openid,pps.read,pps.write,additional_info.projectedProductContext,"
        "additional_info.ownerOrg,uds_read,uds_write,ab.manage,read_organizations,"
        "additional_info.roles,account_cluster.read,creative_production,profile"
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._runner_started = False
        self._stop_event = threading.Event()
        self._profiles: List[RefreshRecord] = []
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._load_profiles()

    def _load_profiles(self):
        with self._lock:
            if not PROFILE_FILE.exists():
                self._profiles = []
                return
            try:
                payload = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
            except Exception:
                self._profiles = []
                return

            profiles = payload.get("profiles") if isinstance(payload, dict) else None
            if not isinstance(profiles, list):
                self._profiles = []
                return

            loaded: List[RefreshRecord] = []
            now_ts = int(time.time())
            for item in profiles:
                try:
                    normalized = self._normalize_stored_profile(item, now_ts)
                except Exception:
                    continue
                loaded.append(normalized)
            self._profiles = loaded

    def _save_profiles(self):
        payload = {
            "version": 2,
            "profiles": self._profiles,
        }
        PROFILE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _validate_bundle(bundle: RefreshRecord) -> RefreshRecord:
        if not isinstance(bundle, dict):
            raise ValueError("bundle must be an object")

        endpoint = bundle.get("endpoint")
        if not isinstance(endpoint, dict):
            raise ValueError("bundle.endpoint is required")

        url = str(endpoint.get("url") or "").strip()
        if not url.startswith(
            "https://adobeid-na1.services.adobe.com/ims/check/v6/token"
        ):
            raise ValueError("invalid endpoint url")

        form = endpoint.get("form")
        headers = endpoint.get("headers")
        if not isinstance(form, dict):
            raise ValueError("bundle.endpoint.form is required")
        if not isinstance(headers, dict):
            raise ValueError("bundle.endpoint.headers is required")

        for key in ("client_id", "scope"):
            if not str(form.get(key) or "").strip():
                raise ValueError(f"bundle form missing {key}")
        if not str(headers.get("Cookie") or "").strip():
            raise ValueError("bundle headers missing Cookie")

        normalized_headers = {
            "Accept": str(headers.get("Accept") or "*/*"),
            "Accept-Language": str(headers.get("Accept-Language") or "en-US,en;q=0.9"),
            "Content-Type": str(
                headers.get("Content-Type")
                or "application/x-www-form-urlencoded;charset=UTF-8"
            ),
            "Cookie": str(headers.get("Cookie") or "").strip(),
            "Origin": str(headers.get("Origin") or "https://firefly.adobe.com"),
            "Referer": str(headers.get("Referer") or "https://firefly.adobe.com/"),
            "User-Agent": str(headers.get("User-Agent") or "Mozilla/5.0"),
        }

        scope = str(form.get("scope") or "").strip()
        scope_parts = [part.strip() for part in scope.split(",") if part.strip()]
        if "profile" not in scope_parts:
            scope_parts.append("profile")

        normalized_form = {
            "client_id": str(form.get("client_id") or "").strip(),
            "guest_allowed": str(form.get("guest_allowed") or "true").strip() or "true",
            "scope": ",".join(scope_parts),
        }

        return {
            "endpoint": {
                "url": url,
                "method": "POST",
                "form": normalized_form,
                "headers": normalized_headers,
            }
        }

    @classmethod
    def _normalize_stored_profile(cls, profile: RefreshRecord, now_ts: int) -> RefreshRecord:
        if not isinstance(profile, dict):
            raise ValueError("invalid profile")
        endpoint = profile.get("endpoint")
        validated = cls._validate_bundle({"endpoint": endpoint})
        profile_id = str(profile.get("id") or "").strip() or uuid.uuid4().hex[:8]
        profile_name = str(profile.get("name") or "").strip()
        if not profile_name:
            profile_name = (
                f"{validated['endpoint']['form']['client_id']}-{profile_id[:4]}"
            )

        state_raw = profile.get("state")
        state: RefreshRecord = state_raw if isinstance(state_raw, dict) else {}
        account_raw = profile.get("account")
        account: RefreshRecord = account_raw if isinstance(account_raw, dict) else {}
        return {
            "id": profile_id,
            "name": profile_name,
            "enabled": bool(profile.get("enabled", True)),
            "cookie_fingerprint": str(
                profile.get("cookie_fingerprint")
                or cls._cookie_fingerprint(
                    str(validated["endpoint"]["headers"].get("Cookie") or "")
                )
            ),
            "imported_at": int(profile.get("imported_at") or now_ts),
            "endpoint": validated["endpoint"],
            "account": {
                "display_name": str(account.get("display_name") or "").strip(),
                "email": str(account.get("email") or "").strip(),
                "user_id": str(account.get("user_id") or "").strip(),
                "source": str(account.get("source") or "").strip(),
                "updated_at": account.get("updated_at"),
            },
            "state": {
                "last_attempt_at": state.get("last_attempt_at"),
                "last_success_at": state.get("last_success_at"),
                "last_error": str(state.get("last_error") or ""),
                "last_http_status": state.get("last_http_status"),
                "next_retry_at": state.get("next_retry_at"),
                "consecutive_failures": int(state.get("consecutive_failures") or 0),
            },
        }

    @staticmethod
    def _format_ts(ts_value) -> str:
        if ts_value is None:
            return "-"
        try:
            dt = datetime.fromtimestamp(float(ts_value))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "-"

    @staticmethod
    def _refresh_interval_hours() -> int:
        raw = config_manager.get("refresh_interval_hours", 15)
        try:
            hours = int(str(raw or "").strip())
        except Exception:
            return 15
        if hours < 1 or hours > 24:
            return 15
        return hours

    @classmethod
    def _refresh_interval_seconds(cls) -> int:
        return cls._refresh_interval_hours() * 3600

    def _requests_proxies(self):
        proxy = str(config_manager.get("proxy", "") or "").strip()
        use_proxy = bool(config_manager.get("use_proxy", False))
        if not (use_proxy and proxy):
            return None
        return {"http": proxy, "https": proxy}

    def _summary_locked(self, profile: RefreshRecord) -> RefreshRecord:
        endpoint_raw = profile.get("endpoint")
        endpoint: RefreshRecord = endpoint_raw if isinstance(endpoint_raw, dict) else {}
        form_raw = endpoint.get("form")
        form: RefreshRecord = form_raw if isinstance(form_raw, dict) else {}
        state_raw = profile.get("state")
        state: RefreshRecord = state_raw if isinstance(state_raw, dict) else {}
        account_raw = profile.get("account")
        account: RefreshRecord = account_raw if isinstance(account_raw, dict) else {}
        return {
            "id": profile.get("id"),
            "name": profile.get("name"),
            "enabled": bool(profile.get("enabled", True)),
            "imported_at": profile.get("imported_at"),
            "endpoint": {
                "url": endpoint.get("url", ""),
                "client_id": form.get("client_id", ""),
            },
            "account": {
                "display_name": str(account.get("display_name") or "").strip(),
                "email": str(account.get("email") or "").strip(),
                "user_id": str(account.get("user_id") or "").strip(),
                "updated_at": account.get("updated_at"),
            },
            "state": {
                **state,
                "next_refresh_at_text": self._format_ts(state.get("next_retry_at")),
                "last_success_at_text": self._format_ts(state.get("last_success_at")),
                "last_attempt_at_text": self._format_ts(state.get("last_attempt_at")),
            },
            "refresh_interval_hours": self._refresh_interval_hours(),
        }

    def list_profiles(self) -> List[RefreshRecord]:
        with self._lock:
            items = [self._summary_locked(p) for p in self._profiles]
        items.sort(key=lambda x: int(x.get("imported_at") or 0), reverse=True)
        return items

    @staticmethod
    def _cookie_string_from_input(cookie_input) -> str:
        if isinstance(cookie_input, str):
            text = cookie_input.strip()
            if text.lower().startswith("cookie:"):
                text = text.split(":", 1)[1].strip()
            return text

        if isinstance(cookie_input, dict):
            if isinstance(cookie_input.get("cookies"), list):
                cookie_input = cookie_input.get("cookies")
            elif isinstance(cookie_input.get("cookie"), (str, list, dict)):
                cookie_input = cookie_input.get("cookie")
            else:
                return ""

        if isinstance(cookie_input, list):
            pairs: List[str] = []
            for item in cookie_input:
                if isinstance(item, str):
                    txt = item.strip()
                    if txt:
                        pairs.append(txt)
                    continue
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                value = str(item.get("value") or "").strip()
                if not name:
                    continue
                pairs.append(f"{name}={value}")
            return "; ".join(pairs)
        return ""

    @staticmethod
    def _normalize_cookie_for_fingerprint(cookie: str) -> str:
        pairs = []
        for part in str(cookie or "").split(";"):
            item = part.strip()
            if item:
                pairs.append(item)
        return "; ".join(sorted(pairs))

    @classmethod
    def _cookie_fingerprint(cls, cookie: str) -> str:
        normalized = cls._normalize_cookie_for_fingerprint(cookie)
        if not normalized:
            return ""
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def import_cookie(self, cookie_input, name: Optional[str] = None) -> RefreshRecord:
        cookie = self._cookie_string_from_input(cookie_input)
        if not cookie:
            raise ValueError("cookie is required")
        validated = self._validate_bundle(
            {
                "endpoint": {
                    "url": self.DEFAULT_REFRESH_URL,
                    "method": "POST",
                    "form": {
                        "client_id": "clio-playground-web",
                        "guest_allowed": "true",
                        "scope": self.DEFAULT_SCOPE,
                    },
                    "headers": {
                        "Accept": "*/*",
                        "Accept-Language": "zh-CN,zh;q=0.9",
                        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                        "Cookie": cookie,
                        "Origin": "https://firefly.adobe.com",
                        "Referer": "https://firefly.adobe.com/",
                        "User-Agent": "Mozilla/5.0",
                    },
                }
            }
        )

        now_ts = int(time.time())
        cookie_fingerprint = self._cookie_fingerprint(cookie)
        profile_id = uuid.uuid4().hex[:8]
        profile_name = str(name or "").strip()
        if not profile_name:
            profile_name = (
                f"{validated['endpoint']['form']['client_id']}-{profile_id[:4]}"
            )

        new_profile = {
            "id": profile_id,
            "name": profile_name,
            "enabled": True,
            "cookie_fingerprint": cookie_fingerprint,
            "imported_at": now_ts,
            "endpoint": validated["endpoint"],
            "account": {
                "display_name": "",
                "email": "",
                "user_id": "",
                "source": "",
                "updated_at": None,
            },
            "state": {
                "last_attempt_at": None,
                "last_success_at": None,
                "last_error": "",
                "last_http_status": None,
                "next_retry_at": time.time() + self._refresh_interval_seconds(),
                "consecutive_failures": 0,
            },
        }

        with self._lock:
            for existing in self._profiles:
                if not cookie_fingerprint:
                    continue
                if str(existing.get("cookie_fingerprint") or "") != cookie_fingerprint:
                    continue
                existing["endpoint"] = validated["endpoint"]
                existing["cookie_fingerprint"] = cookie_fingerprint
                if str(name or "").strip():
                    existing["name"] = str(name or "").strip()
                self._save_profiles()
                return self._summary_locked(existing)

            self._profiles.append(new_profile)
            self._save_profiles()
            return self._summary_locked(new_profile)

    def export_cookies(self, ids: Optional[List[str]] = None) -> List[RefreshRecord]:
        selected_ids = None
        if isinstance(ids, list):
            normalized = [str(x or "").strip() for x in ids]
            selected_ids = {x for x in normalized if x}
        with self._lock:
            out: List[RefreshRecord] = []
            for p in self._profiles:
                pid = str(p.get("id") or "").strip()
                if selected_ids is not None and pid not in selected_ids:
                    continue
                endpoint_raw = p.get("endpoint")
                endpoint: RefreshRecord = (
                    endpoint_raw if isinstance(endpoint_raw, dict) else {}
                )
                headers_raw = endpoint.get("headers")
                headers: RefreshRecord = (
                    headers_raw if isinstance(headers_raw, dict) else {}
                )
                cookie = str(headers.get("Cookie") or "").strip()
                out.append(
                    {
                        "id": pid,
                        "name": str(p.get("name") or "").strip(),
                        "cookie": cookie,
                    }
                )
            return out

    def is_profile_enabled(self, profile_id: str) -> Optional[bool]:
        pid = str(profile_id or "").strip()
        if not pid:
            return None
        with self._lock:
            target = self._find_profile_locked(pid)
            if not target:
                return None
            return bool(target.get("enabled", True))

    def _find_profile_locked(self, profile_id: str) -> Optional[RefreshRecord]:
        for p in self._profiles:
            if p.get("id") == profile_id:
                return p
        return None

    def remove_profile(self, profile_id: str):
        with self._lock:
            target = self._find_profile_locked(profile_id)
            if not target:
                raise KeyError("profile not found")
            self._profiles = [p for p in self._profiles if p.get("id") != profile_id]
            self._save_profiles()
        _token_manager().remove_auto_refresh_by_profile(profile_id)

    def set_enabled(self, profile_id: str, enabled: bool) -> RefreshRecord:
        with self._lock:
            target = self._find_profile_locked(profile_id)
            if not target:
                raise KeyError("profile not found")
            target["enabled"] = bool(enabled)
            state = target.setdefault("state", {})
            if enabled:
                state["next_retry_at"] = time.time() + self._refresh_interval_seconds()
                state["last_error"] = ""
                state["consecutive_failures"] = 0
            self._save_profiles()
            return self._summary_locked(target)

    def _prepare_refresh(
        self, profile_id: str, allow_disabled_profile: bool = False
    ) -> RefreshRecord:
        with self._lock:
            target = self._find_profile_locked(profile_id)
            if not target:
                raise KeyError("profile not found")
            if not allow_disabled_profile and not bool(target.get("enabled", True)):
                raise ValueError("profile is disabled")
            endpoint = target.get("endpoint", {})
            state = target.setdefault("state", {})
            state["last_attempt_at"] = int(time.time())
            snapshot = {
                "id": target.get("id"),
                "name": target.get("name"),
                "url": endpoint.get("url"),
                "headers": dict(endpoint.get("headers") or {}),
                "form": dict(endpoint.get("form") or {}),
            }
            self._save_profiles()
            return snapshot

    def _resolve_account_profile_id(self, profile_id: str, account: RefreshRecord) -> str:
        account_id = str(account.get("user_id") or "").strip()
        if not account_id:
            return str(profile_id or "").strip()

        with self._lock:
            target = self._find_profile_locked(profile_id)
            if not target:
                return str(profile_id or "").strip()

            duplicate = None
            for profile in self._profiles:
                if profile.get("id") == profile_id:
                    continue
                account_raw = profile.get("account")
                profile_account: RefreshRecord = (
                    account_raw if isinstance(account_raw, dict) else {}
                )
                if str(profile_account.get("user_id") or "").strip() == account_id:
                    duplicate = profile
                    break

            if duplicate is None:
                return str(profile_id or "").strip()

            duplicate["endpoint"] = target.get("endpoint", {})
            duplicate["cookie_fingerprint"] = target.get("cookie_fingerprint", "")
            _token_manager().remove_auto_refresh_by_profile(profile_id)
            self._profiles = [p for p in self._profiles if p.get("id") != profile_id]
            self._save_profiles()
            return str(duplicate.get("id") or "").strip()

    def _mark_success(self, profile_id: str, http_status: int):
        with self._lock:
            target = self._find_profile_locked(profile_id)
            if not target:
                return
            state = target.setdefault("state", {})
            state["last_http_status"] = int(http_status)
            state["last_success_at"] = int(time.time())
            state["last_error"] = ""
            state["consecutive_failures"] = 0
            state["next_retry_at"] = time.time() + self._refresh_interval_seconds()
            self._save_profiles()

    def _mark_failure(
        self, profile_id: str, message: str, http_status: Optional[int] = None
    ):
        with self._lock:
            target = self._find_profile_locked(profile_id)
            if not target:
                return
            state = target.setdefault("state", {})
            fails = int(state.get("consecutive_failures", 0)) + 1
            state["consecutive_failures"] = fails
            state["last_error"] = str(message or "")[:500]
            if http_status is not None:
                state["last_http_status"] = int(http_status)
            delays = [60, 180, 600, 1800]
            delay = delays[min(fails - 1, len(delays) - 1)]
            state["next_retry_at"] = time.time() + delay
            self._save_profiles()

    def _fetch_account_info(self, access_token: str) -> RefreshRecord:
        token = str(access_token or "").strip()
        if not token:
            return {}
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        profile_urls = [
            "https://ims-na1.adobelogin.com/ims/profile/v1",
            "https://adobeid-na1.services.adobe.com/ims/profile/v1",
        ]
        for url in profile_urls:
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    timeout=15,
                    proxies=self._requests_proxies(),
                )
            except Exception:
                continue
            if resp.status_code != 200:
                continue
            try:
                data = resp.json()
            except Exception:
                continue
            if not isinstance(data, dict):
                continue

            display_name = str(
                data.get("displayName")
                or data.get("name")
                or data.get("fullName")
                or ""
            ).strip()
            email = str(data.get("email") or "").strip()
            user_id = str(data.get("userId") or data.get("authId") or "").strip()
            if not (display_name or email or user_id):
                continue
            return {
                "display_name": display_name,
                "email": email,
                "user_id": user_id,
                "source": "ims_profile_v1",
                "updated_at": int(time.time()),
            }
        return {}

    @staticmethod
    def _extract_account_id(access_token: str) -> str:
        try:
            payload = _token_manager()._decode_jwt_payload(access_token)
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            return ""
        return str(
            payload.get("user_id") or payload.get("aa_id") or payload.get("sub") or ""
        ).strip()

    def _fetch_credits_balance(self, access_token: str, account_id: str) -> RefreshRecord:
        token = str(access_token or "").strip()
        aid = str(account_id or "").strip()
        if not token:
            raise RuntimeError("empty access token")
        if not aid:
            raise RuntimeError("missing account id")

        resp = requests.get(
            "https://firefly.adobe.io/v1/credits/balance",
            headers={
                "Authorization": f"Bearer {token}",
                "x-api-key": "SunbreakWebUI1",
                "x-account-id": aid,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=20,
            proxies=self._requests_proxies(),
        )
        if resp.status_code != 200:
            raise RuntimeError(f"credits request failed: {resp.status_code}")
        try:
            payload = resp.json()
        except Exception:
            raise RuntimeError("credits response invalid json")
        total_info = payload.get("total", {}) if isinstance(payload, dict) else {}
        quota = total_info.get("quota", {}) if isinstance(total_info, dict) else {}
        return {
            "total": quota.get("total"),
            "used": quota.get("used"),
            "available": quota.get("available"),
            "available_until": total_info.get("availableUntil"),
            "updated_at": int(time.time()),
        }

    def refresh_credits_for_token_id(self, token_id: str) -> RefreshRecord:
        token_info = _token_manager().get_by_id(token_id)
        if not token_info:
            raise KeyError("token not found")
        token_value = str(token_info.get("value") or "").strip()
        account_id = self._extract_account_id(token_value)
        credits = self._fetch_credits_balance(token_value, account_id)
        _token_manager().set_credits(token_id, credits)
        return {
            "token_id": token_id,
            "credits": credits,
        }

    def _set_profile_account(self, profile_id: str, account: RefreshRecord):
        if not account:
            return
        with self._lock:
            target = self._find_profile_locked(profile_id)
            if not target:
                return
            current_raw = target.get("account")
            current: RefreshRecord = (
                current_raw if isinstance(current_raw, dict) else {}
            )
            merged = {
                "display_name": str(
                    account.get("display_name") or current.get("display_name") or ""
                ).strip(),
                "email": str(
                    account.get("email") or current.get("email") or ""
                ).strip(),
                "user_id": str(
                    account.get("user_id") or current.get("user_id") or ""
                ).strip(),
                "source": str(
                    account.get("source") or current.get("source") or ""
                ).strip(),
                "updated_at": account.get("updated_at") or current.get("updated_at"),
            }
            target["account"] = merged
            display_name = merged.get("display_name")
            email = merged.get("email")
            if display_name or email:
                target["name"] = display_name or email
            self._save_profiles()

    def refresh_once(
        self, profile_id: str, allow_disabled_profile: bool = False
    ) -> RefreshRecord:
        snapshot = self._prepare_refresh(
            profile_id, allow_disabled_profile=allow_disabled_profile
        )
        resp = requests.post(
            snapshot["url"],
            headers=snapshot["headers"],
            data=snapshot["form"],
            timeout=30,
            proxies=self._requests_proxies(),
        )

        if resp.status_code != 200:
            self._mark_failure(
                profile_id,
                f"refresh request failed: {resp.status_code} {resp.text[:200]}",
                http_status=resp.status_code,
            )
            raise RuntimeError(
                f"refresh request failed: {resp.status_code} {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except Exception:
            self._mark_failure(
                profile_id,
                "refresh response is not valid json",
                http_status=resp.status_code,
            )
            raise RuntimeError("refresh response is not valid json")

        token = str(data.get("access_token") or "").strip()
        if not token:
            self._mark_failure(
                profile_id,
                "refresh response missing access_token",
                http_status=resp.status_code,
            )
            raise RuntimeError("refresh response missing access_token")

        account = self._fetch_account_info(token)
        target_profile_id = str(snapshot["id"] or "")
        if account:
            target_profile_id = self._resolve_account_profile_id(target_profile_id, account)
            self._set_profile_account(target_profile_id, account)

        profile_name = str(
            account.get("display_name")
            or account.get("email")
            or snapshot["name"]
            or ""
        ).strip()
        profile_email = str(account.get("email") or "").strip()

        token_record = _token_manager().upsert_auto_refresh_token(
            token,
            profile_id=target_profile_id,
            profile_name=profile_name,
            profile_email=profile_email,
        )

        credits_error = ""
        token_id = str(token_record.get("id") or "").strip()
        if token_id:
            try:
                self.refresh_credits_for_token_id(token_id)
            except Exception as exc:
                credits_error = str(exc)
                _token_manager().set_credits_error(token_id, credits_error)

        self._mark_success(target_profile_id, http_status=resp.status_code)

        return {
            "status": "ok",
            "profile_id": target_profile_id,
            "profile_name": profile_name,
            "profile_email": profile_email,
            "expires_in": data.get("expires_in"),
            "credits_error": credits_error,
        }

    def start(self):
        with self._lock:
            if self._runner_started:
                return
            self._runner_started = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    candidates = [
                        {
                            "id": p.get("id"),
                            "enabled": bool(p.get("enabled", True)),
                            "next_retry_at": p.get("state", {}).get("next_retry_at"),
                        }
                        for p in self._profiles
                    ]

                now_ts = time.time()
                for item in candidates:
                    if not item.get("enabled"):
                        continue
                    next_retry = item.get("next_retry_at")
                    if next_retry and now_ts < float(next_retry):
                        continue
                    pid = str(item.get("id") or "")
                    if not pid:
                        continue
                    try:
                        self.refresh_once(pid)
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(2.0)


refresh_manager = RefreshManager()
