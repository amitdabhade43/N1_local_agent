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


def test_person_detail_lookup_can_exit_device_access_clarification():
    request, clarification = native._resolve_device_access_clarification(
        "check PC Admin Rene Piuk user details", history()
    )
    assert request is None
    assert clarification is None


def test_pc_admin_target_is_not_substituted_with_same_name_end_user():
    turns = [
        {"role": "user", "content": "add VM-6804B761 to rene for device access"},
        {"role": "assistant", "content": (
            'Multiple NinjaOne people records matched "rene". '
            'Which user should receive the device access for "VM-6804B761"?\n'
            "1. Rene Piuk <ing.rene.piuk@gmail.com> (end-user ID 144)\n"
            "2. Rene Piuk <rene.piuk@hoerbiger.com> (technician ID 2)\n"
            "3. Rene Piuk <ing.rene.piuk@gmail.com> (console user ID 144)"
        )},
    ]
    request, clarification = native._resolve_device_access_clarification(
        "i want to add in PC Admin Rene Piuk", turns
    )
    assert request is None
    assert "console-user record" in clarification
    assert "does not establish that its role is PC Admin" in clarification
    assert "supports end-user records only" in clarification
    assert "No change was prepared" in clarification


def test_user_not_in_list_can_exit_device_access_clarification():
    request, clarification = native._resolve_device_access_clarification(
        "user not in list", history()
    )
    assert request is None
    assert clarification is None


def test_alternative_target_request_can_exit_device_access_clarification():
    request, clarification = native._resolve_device_access_clarification(
        "i want to add in PC Admin Rene Piuk", history()
    )
    assert request is None
    assert clarification is None


def test_unrelated_count_question_exits_device_access_clarification():
    request, clarification = native._resolve_device_access_clarification(
        "how many user name with rene", history()
    )
    assert request is None
    assert clarification is None


def test_greeting_exits_device_access_clarification():
    request, clarification = native._resolve_device_access_clarification("hi", history())
    assert request is None
    assert clarification is None


def test_no_change_message_exits_device_access_clarification():
    request, clarification = native._resolve_device_access_clarification(
        "No change has been required", history()
    )
    assert request is None
    assert clarification is None


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
            'Which user should receive the device access for "VM-6804B761"?\n'
            "1. Rene Piuk <rene.piuk@example.com>\n"
            "2. Rene Other <rene.other@example.com>\n"
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



def test_selection_parser_rejects_console_user_candidate_for_end_user_only_access_patch():
    turns = [
        {"role": "user", "content": "add VM-6804B761 to rene for device access"},
        {"role": "assistant", "content": (
            'Multiple NinjaOne people records matched "rene". '
            'Which user should receive the device access for "VM-6804B761"?\n'
            "1. Rene Piuk <rene.piuk@example.com> (end-user ID 31)\n"
            "2. Rene Console <rene.console@example.com> (console user ID 88)\n"
            "3. Rene Tech <rene.tech@example.com> (technician ID 89)\n"
            "Reply with the option number, full name, or email address. "
            "No change has been prepared."
        )},
    ]
    request, clarification = native._resolve_device_access_clarification("2", turns)
    assert request is None
    assert clarification is not None
    assert "console user, not an end user" in clarification
    assert "will not substitute a same-name or same-email end-user record" in clarification
    assert "No change was prepared"


def test_system_access_wording_selection_preserves_original_device_and_candidate():
    turns = [
        {"role": "user", "content": "add VM-6804B761 system access to rene"},
        {"role": "assistant", "content": (
            'Multiple NinjaOne people records matched "rene". '
            'Which user should receive the device access for "VM-6804B761"?\n'
            "1. PC Admin Rene Piuk <admin_pr@hoerbiger.com> (end-user ID 31)\n"
            "2. Rene Piuk <ing.rene.piuk@gmail.com> (end-user ID 144)\n"
            "3. Rene Piuk <rene.piuk@hoerbiger.com> (technician ID 2)\n"
            "4. PC Admin Rene Piuk <admin_pr@hoerbiger.com> (console user ID 31)\n"
            "5. Altiris Admin Rene Piuk <aadmin_pr@hoerbiger.com> (technician ID 39)\n"
            "6. Rene Piuk <ing.rene.piuk@gmail.com> (console user ID 144)"
        )},
    ]
    request, clarification = native._resolve_device_access_clarification("1", turns)
    assert clarification is None
    assert request == (
        "add VM-6804B761 to PC Admin Rene Piuk <admin_pr@hoerbiger.com> "
        "for device access"
    )
