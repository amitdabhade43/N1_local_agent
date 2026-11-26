import json

import ninja_agent as core


def test_bulk_device_move_resolves_1000_names_and_prepares_one_approval(monkeypatch):
    names = [f"PC-{index:05d}" for index in range(1000)]
    question = (
        "Please update below devices in **HOERBIGER - Unattended** (ID: 7)\n"
        + "\n".join(names)
    )
    update_op = {
        "id": "updateDevice",
        "method": "PATCH",
        "path": "/v2/device/{id}",
        "summary": "Update device information",
        "request_body": {"media_type": "application/json"},
    }
    list_op = {"id": "getDevices", "method": "GET", "path": "/v2/devices"}
    org_op = {"id": "getOrganizations", "method": "GET", "path": "/v2/organizations"}
    monkeypatch.setattr(core, "OPS", {
        "updateDevice": update_op,
        "getDevices": list_op,
        "getOrganizations": org_op,
    })
    calls = []

    def fake_fetch(operation, path, query):
        calls.append((operation["id"], path, query))
        if operation["id"] == "getOrganizations":
            return [{"id": 7, "name": "HOERBIGER - Unattended"}]
        return [
            {"id": index + 10000, "systemName": name, "organizationId": 6}
            for index, name in enumerate(names)
        ]

    monkeypatch.setattr(core, "fetch", fake_fetch)
    monkeypatch.setattr(core, "_argument_errors", lambda operation, args: [])
    core.clear_pending_session("bulk-1000-test")
    core.set_active_session("bulk-1000-test")
    try:
        result = json.loads(core.recipe_move_devices_to_organization(question))
    finally:
        core.clear_active_session()

    assert result["approval_required"] is True
    assert result["batch"] is True
    assert result["batch_total"] == 1000
    assert len(result["batch_items"]) == 1000
    assert result["batch_items"][0]["device_id"] == 10000
    assert result["batch_items"][-1]["device_id"] == 10999
    assert result["summary"] == "Move 1000 devices to HOERBIGER - Unattended (ID: 7)"
    assert [call[0] for call in calls] == ["getOrganizations", "getDevices"]
    assert calls[1][2] == {"pageSize": 1000}
    core.reject_pending_action(result["action_id"], "bulk-1000-test")


def test_bulk_device_move_does_not_prepare_partial_or_ambiguous_inventory(monkeypatch):
    question = (
        "Move devices to organization HOERBIGER - Unattended (ID: 7)\n"
        "PC-ONE\nPC-MISSING\nPC-DUPLICATE"
    )
    update_op = {
        "id": "updateDevice", "method": "PATCH", "path": "/v2/device/{id}",
        "request_body": {"media_type": "application/json"},
    }
    list_op = {"id": "getDevices", "method": "GET", "path": "/v2/devices"}
    org_op = {"id": "getOrganizations", "method": "GET", "path": "/v2/organizations"}
    monkeypatch.setattr(core, "OPS", {
        "updateDevice": update_op, "getDevices": list_op, "getOrganizations": org_op,
    })
    monkeypatch.setattr(core, "fetch", lambda operation, path, query: (
        [{"id": 7, "name": "HOERBIGER - Unattended"}]
        if operation["id"] == "getOrganizations"
        else [
            {"id": 1, "systemName": "PC-ONE"},
            {"id": 2, "systemName": "PC-DUPLICATE"},
            {"id": 3, "systemName": "PC-DUPLICATE"},
        ]
    ))
    monkeypatch.setattr(core, "_argument_errors", lambda operation, args: [])
    core.clear_pending_session("bulk-partial-test")
    core.set_active_session("bulk-partial-test")
    try:
        result = core.recipe_move_devices_to_organization(question)
    finally:
        core.clear_active_session()

    assert "Not found: PC-MISSING" in result
    assert "Ambiguous names: PC-DUPLICATE" in result
    assert "No changes were prepared." in result
    assert not any(
        item.get("session_id") == "bulk-partial-test"
        for item in core._pending_actions.values()
    )


def test_batch_approval_executes_each_exact_patch_once(monkeypatch):
    op = {"id": "updateDevice", "method": "PATCH", "path": "/v2/device/{id}", "summary": "Update"}
    requests = [
        {"device_name": "PC-A", "path_params": {"id": 11}, "query": {},
         "body": {"organizationId": 7}, "media_type": "application/json"},
        {"device_name": "PC-B", "path_params": {"id": 12}, "query": {},
         "body": {"organizationId": 7}, "media_type": "application/json"},
    ]
    core.clear_pending_session("batch-exec-test")
    core.set_active_session("batch-exec-test")
    try:
        prepared = json.loads(core._prepare_batch_mutation(op, requests, "Move two devices"))
    finally:
        core.clear_active_session()

    calls = []
    monkeypatch.setattr(core, "http_request", lambda method, path, query, body, media_type,
                        headers, approved=False: calls.append((method, path, body, approved)) or {"ok": True})
    result = core.execute_pending_action(prepared["action_id"], "batch-exec-test")
    assert result["ok"] is True
    assert result["total"] == result["succeeded"] == 2
    assert calls == [
        ("PATCH", "/v2/device/11", {"organizationId": 7}, True),
        ("PATCH", "/v2/device/12", {"organizationId": 7}, True),
    ]
    replay = core.execute_pending_action(prepared["action_id"], "batch-exec-test")
    assert replay["ok"] is False
    assert len(calls) == 2
