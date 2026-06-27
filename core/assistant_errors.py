from __future__ import annotations


class AssistantClientError(Exception):
    pass


class AssistantHttpError(AssistantClientError):
    def __init__(self, message: str, *, status_code: int):
        super().__init__(message)
        self.status_code = status_code
