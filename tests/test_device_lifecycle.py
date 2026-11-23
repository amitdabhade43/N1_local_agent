import ninja_agent


def _activity(activity_id, device_id, timestamp):
    return {
        "id": activity_id,
        "activityTime": timestamp,
        "deviceId": device_id,
        "statusCode": "NODE_DELETED",
        "message": "Device 'B82ZST3' deleted.",
        "data": {
            "message": {
                "params": {
                    "appUserId": 2,
                    "appUserName": "Rene Piuk",
                    "appUserEmail": "rene.piuk@hoerbiger.com",
                    "organizationName": "HOERBIGER",
                }
            }
        },
    }


def test_device_lifecycle_reports_deleted_registrations_and_current_registration(monkeypatch):
    operations = {
        "/v2/activities": {
            "params": [
                {"name": "pageSize"}, {"name": "newerThan"},
                {"name": "status"}, {"name": "class"},
            ]
        },
        "/v2/devices": {"params": []},
    }
    monkeypatch.setattr(ninja_agent, "op_by_path", lambda path: operations.get(path))
    monkeypatch.setattr(
        ninja_agent,
        "fetch",
        lambda op, path, query: ninja_agent.FetchRows([
            {
                "id": 8100,
                "systemName": "B82ZST3",
                "organizationName": "HOERBIGER",
                "serialNumber": "SERIAL-8100",
            }
        ]),
    )
    calls = []

    def fake_http_get(path, query):
        calls.append((path, dict(query)))
        return {
            "activities": [
                _activity(8200151, 5988, 1795427226.754357),
                _activity(8199940, 6325, 1795426833.842396),
            ]
        }

    monkeypatch.setattr(ninja_agent, "http_get", fake_http_get)
    answer = ninja_agent.recipe_device_lifecycle("Who deleted B82ZST3?")

    assert "Current record ID **8100**" in answer
    assert "Historical deletion events:** 2" in answer
    assert "Activity **8199940**" in answer
    assert "node ID **6325**" in answer
    assert "Activity **8200151**" in answer
    assert "node ID **5988**" in answer
    assert "Rene Piuk (rene.piuk@hoerbiger.com)" in answer
    assert "may represent repeated registrations" in answer
    assert calls[0][1]["status"] == "NODE_DELETED"
    assert calls[0][1]["pageSize"] == 1000


def test_device_lifecycle_does_not_treat_current_inventory_as_deletion_history(monkeypatch):
    operations = {
        "/v2/activities": {
            "params": [
                {"name": "pageSize"}, {"name": "newerThan"},
                {"name": "status"}, {"name": "class"},
            ]
        },
        "/v2/devices": {"params": []},
    }
    monkeypatch.setattr(ninja_agent, "op_by_path", lambda path: operations.get(path))
    monkeypatch.setattr(
        ninja_agent,
        "fetch",
        lambda op, path, query: ninja_agent.FetchRows([
            {"id": 8100, "systemName": "B82ZST3", "offline": False}
        ]),
    )
    monkeypatch.setattr(ninja_agent, "http_get", lambda path, query: {"activities": []})

    answer = ninja_agent.recipe_device_lifecycle("Who deleted B82ZST3?")

    assert "Current record ID **8100**" in answer
    assert "no matching deletion event was found" in answer
    assert "This is not proof of absence" in answer


def test_device_lifecycle_refuses_to_claim_complete_history_when_inventory_or_audit_fails(monkeypatch):
    operations = {
        "/v2/activities": {
            "params": [
                {"name": "pageSize"}, {"name": "newerThan"},
                {"name": "status"}, {"name": "class"},
            ]
        },
        "/v2/devices": {"params": []},
    }
    monkeypatch.setattr(ninja_agent, "op_by_path", lambda path: operations.get(path))
    monkeypatch.setattr(
        ninja_agent,
        "fetch",
        lambda op, path, query: ninja_agent.FetchRows(
            [{"id": 8100, "systemName": "B82ZST3"}],
            complete=False,
            note="Only one page was returned.",
        ),
    )
    monkeypatch.setattr(
        ninja_agent,
        "http_get",
        lambda path, query: (_ for _ in ()).throw(TimeoutError("audit timeout")),
    )

    answer = ninja_agent.recipe_device_lifecycle("Who deleted B82ZST3?")

    assert "audit request failed before the search completed" in answer
    assert "history may be incomplete" in answer


def test_device_lifecycle_uses_documented_cursor_to_search_older_activity_pages(monkeypatch):
    operations = {
        "/v2/activities": {
            "params": [
                {"name": "pageSize"}, {"name": "newerThan"},
                {"name": "status"}, {"name": "class"},
            ]
        },
        "/v2/devices": {"params": []},
    }
    monkeypatch.setattr(ninja_agent, "op_by_path", lambda path: operations.get(path))
    monkeypatch.setattr(
        ninja_agent,
        "fetch",
        lambda op, path, query: ninja_agent.FetchRows([
            {"id": 8100, "systemName": "B82ZST3", "organizationName": "HOERBIGER"}
        ]),
    )
    calls = []

    def fake_http_get(path, query):
        calls.append(dict(query))
        if "newerThan" not in query:
            return {
                "activities": [
                    {
                        "id": 9000 - index,
                        "statusCode": "NODE_DELETED",
                        "message": f"Device 'OTHER-{index}' deleted.",
                    }
                    for index in range(1000)
                ]
            }
        return {"activities": [_activity(7999, 5988, 1795427226.754357)]}

    monkeypatch.setattr(ninja_agent, "http_get", fake_http_get)
    answer = ninja_agent.recipe_device_lifecycle("Who deleted B82ZST3?")

    assert len(calls) == 2
    assert calls[1]["newerThan"] == 8001
    assert "Activity **7999**" in answer
    assert "node ID **5988**" in answer


def test_device_lifecycle_reads_actor_from_activity_user_details():
    actor, actor_id = ninja_agent._device_lifecycle_actor({
        "userDetails": {
            "id": 2,
            "name": "Rene Piuk",
            "email": "rene.piuk@hoerbiger.com",
        }
    })
    assert actor == "Rene Piuk (rene.piuk@hoerbiger.com)"
    assert actor_id == 2


def test_fast_path_routes_deletion_question_to_lifecycle_recipe(monkeypatch):
    monkeypatch.setattr(ninja_agent, "recipe_latest_end_user", lambda *args, **kwargs: None)
    monkeypatch.setattr(ninja_agent, "recipe_count_technician_role_users", lambda *args, **kwargs: None)
    monkeypatch.setattr(ninja_agent, "recipe_device_lifecycle", lambda question: "lifecycle evidence")
    assert ninja_agent.fast_path("Who deleted B82ZST3?") == "lifecycle evidence"
