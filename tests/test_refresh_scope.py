import unittest

from core.refresh_mgr import RefreshManager


class RefreshScopeTests(unittest.TestCase):
    def test_refresh_bundle_adds_chat_platform_scopes(self):
        validated = RefreshManager._validate_bundle(
            {
                "endpoint": {
                    "url": RefreshManager.DEFAULT_REFRESH_URL,
                    "form": {
                        "client_id": "clio-playground-web",
                        "guest_allowed": "true",
                        "scope": "AdobeID,firefly_api,profile",
                    },
                    "headers": {
                        "Cookie": "a=b",
                    },
                }
            }
        )

        scope = validated["endpoint"]["form"]["scope"]
        parts = [part.strip() for part in scope.split(",")]
        self.assertIn("profile", parts)
        self.assertIn("tk_platform", parts)
        self.assertIn("tk_platform_sync", parts)


if __name__ == "__main__":
    _ = unittest.main()
