import json

import pytest

import ninja_agent as core


@pytest.mark.parametrize("question,expected", [
    # one per line; names that merely CONTAIN instruction words must be kept
    ("Move these devices to organization Acme (ID: 7)\nPC-001\nDEVICE-12\nUPDATE-SRV01\nORG-LAP-1\nTARGET-PC",
     ["PC-001", "DEVICE-12", "UPDATE-SRV01", "ORG-LAP-1", "TARGET-PC"]),
    # comma list: the first item is glued to the sentence prefix and must not be lost
    ("Please move devices to organization Acme (ID: 7): PC-001, DEVICE-12, ORG-LAP-1, PC-002",
     ["PC-001", "DEVICE-12", "ORG-LAP-1", "PC-002"]),
    ("Move devices: PC-001, PC-002 to organization Acme (ID: 7)", ["PC-001", "PC-002"]),
    ("Update devices in Acme (ID: 7)\n- PC-001\n- PC-002\n3. PC-003", ["PC-001", "PC-002", "PC-003"]),
    # headers and prose are never device names
    ("Devices:\nPC-001\nPC-002", ["PC-001", "PC-002"]),
    ("Please update below devices in Acme (ID: 7)\nPC-001", ["PC-001"]),
    # duplicates are preserved so the resolver can report them
    ("Move devices to org Acme (ID: 7)\nPC-001\nPC-001", ["PC-001", "PC-001"]),
    # a device literally called UPDATE is a name, not an instruction
    ("Move devices to org Acme (ID: 7)\nUPDATE\nPC-001", ["UPDATE", "PC-001"]),
])
def test_names_are_extracted_without_silent_drops(question, expected):
    assert core._bulk_device_names_from_request(question) == expected


def _setup(monkeypatch, devices, target_id=7):
    update_op = {"id": "updateDevice", "method": "PATCH", "path": "/v2/device/{id}",
                 "summary": "Update", "request_body": {"media_type": "application/json"}}
    monkeypatch.setattr(core, "OPS", {
        "updateDevice": update_op,
        "getDevices": {"id": "getDevices", "method": "GET", "path": "/v2/devices"},
        "getOrganizations": {"id": "getOrganizations", "method": "GET", "path": "/v2/organizations"},
    })

    def fake_fetch(operation, path, query):
        if operation["id"] == "getOrganizations":
            return [{"id": target_id, "name": "Acme - Unattended"}]
        return devices

    monkeypatch.setattr(core, "fetch", fake_fetch)
    monkeypatch.setattr(core, "_argument_errors", lambda operation, args: [])


def _run(question, session="bulk-names-test"):
    core.clear_pending_session(session)
    core.set_active_session(session)
    try:
        return core.recipe_move_devices_to_organization(question)
    finally:
        core.clear_active_session()


def test_devices_already_in_target_organization_are_not_counted_as_moves(monkeypatch):
    _setup(monkeypatch, [
        {"id": 1, "systemName": "PC-001", "organizationId": 6},
        {"id": 2, "systemName": "PC-002", "organizationId": 7},
    ])
    out = json.loads(_run("Move devices to organization Acme - Unattended (ID: 7)\nPC-001\nPC-002"))
    assert out["batch_total"] == 1
    assert [i["device_name"] for i in out["batch_items"]] == ["PC-001"]
    assert out["summary"] == "Move 1 devices to Acme - Unattended (ID: 7)"
    assert [i["name"] for i in out["skipped_items"]] == ["PC-002"]
    assert "Already in the target organization" in out["skipped_items"][0]["reason"]
    core.reject_pending_action(out["action_id"], "bulk-names-test")


def test_everything_already_in_target_prepares_nothing(monkeypatch):
    _setup(monkeypatch, [{"id": 2, "systemName": "PC-002", "organizationId": 7}])
    text = _run("Move devices to organization Acme - Unattended (ID: 7)\nPC-002")
    assert "Nothing was prepared" in text and "Already in the target organization" in text


def test_non_numeric_device_id_is_skipped_not_a_crash(monkeypatch):
    _setup(monkeypatch, [
        {"id": "abc", "systemName": "PC-001", "organizationId": 6},
        {"id": 2, "systemName": "PC-002", "organizationId": 6},
    ])
    out = json.loads(_run("Move devices to organization Acme - Unattended (ID: 7)\nPC-001\nPC-002"))
    assert [i["device_name"] for i in out["batch_items"]] == ["PC-002"]
    assert "numeric device ID" in out["skipped_items"][0]["reason"]
    core.reject_pending_action(out["action_id"], "bulk-names-test")


def test_hyphenated_keyword_names_resolve_end_to_end(monkeypatch):
    _setup(monkeypatch, [
        {"id": 1, "systemName": "DEVICE-12", "organizationId": 6},
        {"id": 2, "systemName": "ORG-LAP-1", "organizationId": 6},
        {"id": 3, "systemName": "PC-001", "organizationId": 6},
    ])
    out = json.loads(_run(
        "Please move devices to organization Acme - Unattended (ID: 7): PC-001, DEVICE-12, ORG-LAP-1"))
    assert out["batch_total"] == 3 and out["requested_total"] == 3
    core.reject_pending_action(out["action_id"], "bulk-names-test")


def test_verb_glued_to_first_item_without_colon():
    q = "Move PC-001, PC-002 to organization Acme (ID: 7)"
    assert core._bulk_device_names_from_request(q) == ["PC-001", "PC-002"]
