import json

import ninja_agent


def test_technician_role_count_counts_unique_members_not_duplicate_role_memberships(monkeypatch):
    operation = {
        "id": "getUserRoles",
        "method": "GET",
        "path": "/v2/user/roles",
    }
    rows = ninja_agent.FetchRows([
        {
            "id": 10,
            "name": "Technician Admin",
            "type": "TECHNICIAN",
            "members": [
                {"id": 1, "name": "Rene Piuk"},
                {"id": 2, "name": "Alex Example"},
            ],
        },
        {
            "id": 11,
            "name": "Technician Support",
            "type": "TECHNICIAN",
            "members": [
                {"id": 2, "name": "Alex Example"},
                {"id": 3, "name": "Sam Example"},
            ],
        },
    ])
    calls = []

    monkeypatch.setattr(
        ninja_agent,
        "op_by_path",
        lambda path: operation if path == "/v2/user/roles" else None,
    )

    def fake_fetch(op, path, query):
        calls.append((op, path, query))
        return rows

    monkeypatch.setattr(ninja_agent, "fetch", fake_fetch)

    answer = ninja_agent.recipe_count_technician_role_users(
        "How many user in technician role?"
    )

    assert "3 distinct user(s)" in answer
    assert "2 technician-type role(s)" in answer
    assert "4 role membership(s)" in answer
    assert calls == [(operation, "/v2/user/roles", {"roleType": "TECHNICIAN"})]


def test_technician_role_count_is_used_by_native_fast_path(monkeypatch):
    expected = "3 distinct user(s) are members of 2 technician-type role(s)."
    called = []

    monkeypatch.setattr(
        ninja_agent,
        "recipe_count_technician_role_users",
        lambda question: called.append(question) or expected,
    )

    answer = ninja_agent.fast_path("How many user in technician role?")

    assert answer == expected
    assert called == ["How many user in technician role?"]


def test_technician_role_count_does_not_claim_complete_total_for_partial_data(monkeypatch):
    operation = {
        "id": "getUserRoles",
        "method": "GET",
        "path": "/v2/user/roles",
    }
    rows = ninja_agent.FetchRows(
        [{"id": 10, "name": "Technician Admin", "members": [{"id": 1}]}],
        complete=False,
        note="Role inventory paging did not complete.",
    )
    monkeypatch.setattr(
        ninja_agent,
        "op_by_path",
        lambda path: operation if path == "/v2/user/roles" else None,
    )
    monkeypatch.setattr(ninja_agent, "fetch", lambda *args, **kwargs: rows)

    answer = ninja_agent.recipe_count_technician_role_users(
        "How many users are in technician roles?"
    )

    assert answer.startswith("At least 1 distinct user")
    assert "complete total is unknown" in answer
    assert "paging did not complete" in answer


def test_person_first_device_access_request_resolves_user_and_device_separately(monkeypatch):
    end_user = {
        "id": 42,
        "firstName": "Rene",
        "lastName": "Piuk",
        "email": "rene.piuk@example.com",
        "userType": "END_USER",
        "organizationId": 6,
        "accessibleDeviceIds": [],
    }
    device = {
        "id": 50,
        "systemName": "VM-6804B761",
        "organizationId": 7,
    }
    calls = []

    def fake_resolve(arguments):
        kind, name = arguments["kind"], arguments["name"]
        calls.append((kind, name))
        if kind == "end_user":
            return json.dumps({"matches": [end_user], "total_matches": 1})
        if kind == "user":
            # The same END_USER also appears in /v2/users; this must not
            # be misclassified as a second console-user identity.
            return json.dumps({"matches": [end_user], "total_matches": 1})
        if kind in {"technician", "contact"}:
            return json.dumps({"matches": [], "total_matches": 0})
        if kind == "device":
            return json.dumps({"matches": [device], "total_matches": 1})
        return json.dumps({"matches": [], "total_matches": 0})

    monkeypatch.setattr(ninja_agent, "t_resolve", fake_resolve)
    monkeypatch.setattr(ninja_agent, "OPS", {
        "patchEndUserDeviceAccess": {
            "id": "patchEndUserDeviceAccess",
            "method": "PATCH",
            "path": "/v2/user/end-user/{id}/device-access",
        }
    })
    prepared_calls = []

    def fake_t_call(arguments):
        prepared_calls.append(arguments)
        return json.dumps({
            "approval_required": True,
            "operation_id": arguments["operation_id"],
            "method": "PATCH",
            "path": "/v2/user/end-user/42/device-access",
            "body": arguments["body"],
        })

    monkeypatch.setattr(ninja_agent, "t_call", fake_t_call)

    result = json.loads(
        ninja_agent.recipe_add_device_access("Give Rene Piuk access to VM-6804B761")
    )

    assert result["approval_required"] is True
    assert result["body"] == {"addDeviceIds": [50]}
    assert "Rene Piuk" in result["summary"]
    assert "VM-6804B761" in result["summary"]
    # A full name is identity-qualified: resolve that exact end-user first,
    # then resolve the device separately. No first-name cross-inventory scan.
    assert calls == [
        ("end_user", "Rene Piuk"),
        ("device", "VM-6804B761"),
    ]
    assert len(prepared_calls) == 1


def test_person_first_device_access_request_stops_for_ambiguous_rene(monkeypatch):
    end_user = {
        "id": 42,
        "firstName": "Rene",
        "lastName": "Piuk",
        "email": "rene.piuk@example.com",
        "userType": "END_USER",
        "accessibleDeviceIds": [],
    }
    technician = {
        "id": 31,
        "firstName": "Rene",
        "lastName": "Piuk",
        "email": "pc.admin.rene@example.com",
        "userType": "TECHNICIAN",
    }

    def fake_resolve(arguments):
        kind = arguments["kind"]
        if kind == "end_user":
            return json.dumps({"matches": [end_user], "total_matches": 1})
        if kind == "user":
            return json.dumps({
                "matches": [end_user, technician],
                "total_matches": 2,
            })
        if kind == "technician":
            return json.dumps({"matches": [technician], "total_matches": 1})
        if kind == "contact":
            return json.dumps({"matches": [], "total_matches": 0})
        raise AssertionError("Device lookup must wait until the person is selected.")

    monkeypatch.setattr(ninja_agent, "t_resolve", fake_resolve)
    monkeypatch.setattr(ninja_agent, "t_call", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("No write may be prepared for an ambiguous person.")
    ))

    answer = ninja_agent.recipe_add_device_access("Give Rene Piuk access to VM-6804B761")

    assert "Multiple NinjaOne people records matched" in answer
    assert "Which user should receive the device access" in answer
    assert "No change has been prepared" in answer
