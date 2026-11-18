"""Tests for Native device-access clarification and safe request resumption."""

import json
from unittest.mock import patch

import ninja_native as native
import ninja_agent as core


def history():
    return [
        {"role": "user", "content": "add VM-6804B761 to rene for device access"},
        {"role": "assistant", "content": (
            'Multiple end-user records matched "rene". Which user should receive '
            'the device access for "VM-6804B761"?\n'
            "1. Rene Piuk <rene.piuk@example.com> (end-user ID 31)\n"
            "2. Rene Other <rene.other@example.com> (end-user ID 32)\n"
            "Reply with the option number, full name, or email address. "
            "No change has been prepared; I will prepare the approval only after you choose one."
        )},
    ]


def test_selection_by_number_reconstructs_original_device_access_request():
    request, clarification = native._resolve_device_access_clarification("1", history())
    assert clarification is None
    assert request == "add VM-6804B761 to Rene Piuk <rene.piuk@example.com> for device access"


def test_selection_by_exact_email_reconstructs_original_request():
    request, clarification = native._resolve_device_access_clarification(
        "rene.other@example.com", history()
    )
    assert clarification is None
    assert request == "add VM-6804B761 to Rene Other <rene.other@example.com> for device access"


def test_non_unique_or_invalid_reply_asks_again_without_preparing_write():
    request, clarification = native._resolve_device_access_clarification("Rene", history())
    assert request is None
    assert "couldn't match that reply" in clarification
    assert "1. Rene Piuk" in clarification
    assert "2. Rene Other" in clarification
    assert "No change has been prepared" in clarification



def test_valid_selection_still_works_after_an_invalid_clarification_reply():
    turns = history() + [
        {"role": "user", "content": "Rene"},
        {"role": "assistant", "content": (
            "I couldn't match that reply to one listed user. "
            'Which user should receive the device access for "VM-6804B761"?\\n'
            "1. Rene Piuk <rene.piuk@example.com>\\n"
            "2. Rene Other <rene.other@example.com>\\n"
            "No change has been prepared."
        )},
    ]
    request, clarification = native._resolve_device_access_clarification("2", turns)
    assert clarification is None
    assert request == "add VM-6804B761 to Rene Other <rene.other@example.com> for device access"


def test_selection_by_full_name_is_supported():
    request, clarification = native._resolve_device_access_clarification("Rene Piuk", history())
    assert clarification is None
    assert request == "add VM-6804B761 to Rene Piuk <rene.piuk@example.com> for device access"


def test_selection_resumes_recipe_and_returns_approval_without_llm():
    session_id = "test-device-access-choice"
    native._sessions[session_id] = history()
    prepared = {
        "approval_required": True,
        "summary": "Add VM-6804B761 to Rene Piuk",
        "operation_id": "patchEndUserDeviceAccess",
        "method": "PATCH",
        "path": "/v2/user/end-user/31/device-access",
        "body": {"addDeviceIds": [50]},
    }
    try:
        with patch.object(native, "_ensure"), \
             patch.object(core, "recipe_add_device_access", return_value=json.dumps(prepared)) as recipe:
            result = native._try_fast_path(session_id, "1")
        assert "Approval required" in result[0]
        assert any(item["kind"] == "approval_required" for item in result[1])
        assert recipe.call_args.args[0] == (
            "add VM-6804B761 to Rene Piuk <rene.piuk@example.com> for device access"
        )
    finally:
        native._sessions.pop(session_id, None)
