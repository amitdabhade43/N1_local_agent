"""Regression tests for natural-language Native device-access approvals."""

import json
from unittest.mock import patch

import ninja_agent as core


DEVICE = {
    "id": 50,
    "systemName": "VM-6804B761",
    "organizationId": 7,
}
OPERATION = {
    "id": "patchEndUserDeviceAccess",
    "method": "PATCH",
    "path": "/v2/user/end-user/{id}/device-access",
}


def person(user_id, first, last, email):
    return {
        "id": user_id,
        "firstName": first,
        "lastName": last,
        "email": email,
        "accessibleDeviceIds": [],
        "organizationId": 7,
        "match": 1.0,
    }


def resolver_for(users, other_people=None):
    other_people = other_people or {}

    def resolve(args):
        kind, name = args["kind"], args["name"]
        if kind == "end_user":
            return json.dumps({"matches": users})
        if kind == "device" and name == "VM-6804B761":
            return json.dumps({"matches": [DEVICE]})
        return json.dumps({"matches": other_people.get(kind, [])})

    return resolve


def approval():
    return json.dumps({
        "approval_required": True,
        "action_id": "approval-device-access",
        "operation_id": OPERATION["id"],
        "method": "PATCH",
        "path": "/v2/user/end-user/31/device-access",
        "body": {"addDeviceIds": [50]},
    })


def test_conversational_device_access_request_prepares_approval():
    rene = person(31, "Rene", "Piuk", "rene.piuk@example.com")
    with patch.object(core, "OPS", {OPERATION["id"]: OPERATION}), \
         patch.object(core, "t_resolve", side_effect=resolver_for([rene])), \
         patch.object(core, "t_call", return_value=approval()) as call:
        result = json.loads(core.recipe_add_device_access(
            "add VM-6804B761 to rene for device access"
        ))

    assert result["approval_required"] is True
    assert call.call_count == 1
    assert call.call_args.args[0]["operation_id"] == OPERATION["id"]
    assert call.call_args.args[0]["path_params"] == {"id": 31}
    assert call.call_args.args[0]["body"] == {"addDeviceIds": [50]}
    assert "Rene Piuk" in result["summary"]


def test_first_name_matching_multiple_end_users_requires_clarification():
    users = [
        person(31, "Rene", "Piuk", "rene.piuk@example.com"),
        person(32, "Rene", "Other", "rene.other@example.com"),
    ]
    with patch.object(core, "t_resolve", side_effect=resolver_for(users)), \
         patch.object(core, "t_call") as call:
        result = core.recipe_add_device_access(
            "add VM-6804B761 to rene for device access"
        )

    assert "Multiple NinjaOne people records matched" in result
    assert "1. Rene Piuk <rene.piuk@example.com>" in result
    assert "2. Rene Other <rene.other@example.com>" in result
    assert "Which user should receive the device access" in result
    assert "Reply with the option number, full name, or email address" in result
    assert "No change has been prepared" in result
    call.assert_not_called()


def test_non_end_user_match_is_reported_without_preparing_a_write():
    console_user = person(88, "Rene", "Piuk", "rene.piuk@example.com")
    with patch.object(core, "t_resolve", side_effect=resolver_for(
        [], {"user": [console_user]}
    )), patch.object(core, "t_call") as call:
        result = core.recipe_add_device_access(
            "add VM-6804B761 to rene for device access"
        )

    assert "console user" in result
    assert "not an end-user record" in result
    assert "supports end-user records only" in result
    assert "No change was prepared" in result
    call.assert_not_called()



def test_single_end_user_is_not_auto_selected_when_other_people_categories_match():
    end_user = person(31, "Rene", "Piuk", "rene.piuk@example.com")
    console_user = person(88, "Rene", "Piuk", "rene.console@example.com")
    technician = person(89, "Rene", "Piuk", "rene.tech@example.com")
    contact = person(90, "Rene", "Piuk", "rene.contact@example.com")
    with patch.object(core, "t_resolve", side_effect=resolver_for(
        [end_user],
        {"user": [console_user], "technician": [technician], "contact": [contact]},
    )), patch.object(core, "t_call") as call:
        result = core.recipe_add_device_access(
            "add VM-6804B761 to rene for device access"
        )

    assert "Multiple NinjaOne people records matched" in result
    assert "1. Rene Piuk <rene.piuk@example.com> (end-user ID 31)" in result
    assert "(console user ID 88)" in result
    assert "(technician ID 89)" in result
    assert "(contact ID 90)" in result
    assert "Only an end-user record supports" in result
    assert "No change has been prepared" in result
    call.assert_not_called()


def test_first_name_search_includes_token_inside_compound_pc_admin_name():
    regular_end_user = person(144, "Rene", "Piuk", "ing.rene.piuk@gmail.com")
    pc_admin_end_user = person(31, "PC Admin", "Rene Piuk", "admin_pr@hoerbiger.com")
    console_user = person(144, "PC Admin", "Rene Piuk", "ing.rene.piuk@gmail.com")
    technician = person(2, "Rene", "Piuk", "rene.piuk@hoerbiger.com")
    technician["userType"] = "TECHNICIAN"

    with patch.object(core, "t_resolve", side_effect=resolver_for(
        [regular_end_user, pc_admin_end_user],
        {"user": [console_user], "technician": [technician]},
    )), patch.object(core, "t_call") as call:
        result = core.recipe_add_device_access(
            "Add VM-6804B761 to Rene for device access"
        )

    assert "Multiple NinjaOne people records matched" in result
    assert "(end-user ID 144)" in result
    assert "PC Admin Rene Piuk <admin_pr@hoerbiger.com> (end-user ID 31)" in result
    assert "(technician ID 2)" in result
    assert "(console user ID 144)" in result
    assert result.count("PC Admin Rene Piuk <admin_pr@hoerbiger.com>") == 1
    call.assert_not_called()


def test_technician_duplicate_returned_by_user_and_technician_search_is_deduplicated():
    end_user = person(31, "Rene", "Piuk", "rene.piuk@example.com")
    technician = person(89, "Rene", "Piuk", "rene.tech@example.com")
    technician["userType"] = "TECHNICIAN"
    with patch.object(core, "t_resolve", side_effect=resolver_for(
        [end_user],
        {"user": [technician], "technician": [technician]},
    )), patch.object(core, "t_call") as call:
        result = core.recipe_add_device_access(
            "add VM-6804B761 to rene for device access"
        )

    assert "Multiple NinjaOne people records matched" in result
    assert result.count("Rene Piuk <rene.tech@example.com>") == 1
    assert "(technician ID 89)" in result
    call.assert_not_called()


def test_incomplete_people_inventory_fails_closed_even_with_one_end_user_match():
    end_user = person(31, "Rene", "Piuk", "rene.piuk@example.com")

    def resolve(args):
        if args["kind"] == "end_user":
            return json.dumps({"matches": [end_user]})
        if args["kind"] == "device":
            return json.dumps({"matches": [DEVICE]})
        if args["kind"] == "contact":
            return json.dumps({
                "matches": [],
                "partial": True,
                "pagination_note": "Contact results were limited to one page.",
            })
        return json.dumps({"matches": []})

    with patch.object(core, "t_resolve", side_effect=resolve), \
         patch.object(core, "t_call") as call:
        result = core.recipe_add_device_access(
            "add VM-6804B761 to rene for device access"
        )

    assert "inventory search is incomplete" in result
    assert "No person was selected" in result
    assert "No change was prepared" in result
    call.assert_not_called()


def test_exact_email_still_selects_the_end_user_when_other_names_are_similar():
    end_user = person(31, "Rene", "Piuk", "rene.piuk@example.com")
    console_user = person(88, "Rene", "Piuk", "rene.console@example.com")
    with patch.object(core, "OPS", {OPERATION["id"]: OPERATION}), \
         patch.object(core, "t_resolve", side_effect=resolver_for(
             [end_user], {"user": [console_user]}
         )), patch.object(core, "t_call", return_value=approval()) as call:
        result = json.loads(core.recipe_add_device_access(
            "add VM-6804B761 to <rene.piuk@example.com> for device access"
        ))

    assert result["approval_required"] is True
    assert call.call_count == 1
    assert call.call_args.args[0]["path_params"] == {"id": 31}



def test_exact_email_collision_with_technician_does_not_prepare_write():
    end_user = person(31, "Rene", "Piuk", "rene.piuk@example.com")
    technician = person(89, "Rene", "Piuk", "rene.piuk@example.com")
    technician["userType"] = "TECHNICIAN"
    with patch.object(core, "OPS", {OPERATION["id"]: OPERATION}), \
         patch.object(core, "t_resolve", side_effect=resolver_for(
             [end_user], {"user": [technician], "technician": [technician]}
         )), patch.object(core, "t_call") as call:
        result = core.recipe_add_device_access(
            "add VM-6804B761 to <rene.piuk@example.com> for device access"
        )

    assert "Multiple NinjaOne people records matched" in result
    assert "(end-user ID 31)" in result
    assert "(technician ID 89)" in result
    assert "No change has been prepared" in result
    call.assert_not_called()


def test_people_inventory_api_error_fails_closed():
    end_user = person(31, "Rene", "Piuk", "rene.piuk@example.com")

    def resolve(args):
        if args["kind"] == "end_user":
            return json.dumps({"matches": [end_user]})
        if args["kind"] == "user":
            return json.dumps({"error": "403 Forbidden"})
        return json.dumps({"matches": []})

    with patch.object(core, "t_resolve", side_effect=resolve), \
         patch.object(core, "t_call") as call:
        result = core.recipe_add_device_access(
            "add VM-6804B761 to rene for device access"
        )

    assert "403 Forbidden" in result
    assert "identity search is incomplete" in result
    assert "No change was prepared" in result
    call.assert_not_called()
