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

    assert "Multiple end-user records matched" in result
    assert "Rene Piuk" in result
    assert "Rene Other" in result
    assert "No change was prepared" in result
    call.assert_not_called()


def test_non_end_user_match_is_reported_without_preparing_a_write():
    console_user = person(88, "Rene", "Piuk", "rene.piuk@example.com")
    with patch.object(core, "t_resolve", side_effect=resolver_for(
        [], {"user": [console_user]}
    )), patch.object(core, "t_call") as call:
        result = core.recipe_add_device_access(
            "add VM-6804B761 to rene for device access"
        )

    assert "No end-user record matched" in result
    assert "user" in result
    assert "supports end-user records only" in result
    assert "No change was prepared" in result
    call.assert_not_called()
