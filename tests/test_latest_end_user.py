import ninja_agent


def test_latest_end_user_uses_highest_id_from_complete_inventory(monkeypatch):
    rows = ninja_agent.FetchRows([
        {"id": 185, "firstName": "Alexander", "lastName": "Dinkel", "email": "alex@example.com"},
        {"id": 122, "firstName": "Older", "lastName": "User"},
        {"id": "179", "firstName": "Middle", "lastName": "User"},
    ])
    monkeypatch.setattr(ninja_agent, "op_by_path", lambda path: {"id": "getEndUsers"} if path == "/v2/user/end-users" else None)
    monkeypatch.setattr(ninja_agent, "fetch", lambda op, path, query: rows)

    answer = ninja_agent.recipe_latest_end_user("who is the last end user created?")
    assert "Alexander Dinkel" in answer
    assert "ID 185" in answer
    assert "alex@example.com" in answer
    assert "all 3 returned records" in answer
    assert "does not expose a creation timestamp" in answer


def test_latest_end_user_handles_common_typo_and_refuses_partial_inventory(monkeypatch):
    rows = ninja_agent.FetchRows(
        [{"id": 185, "firstName": "Alexander", "lastName": "Dinkel"}],
        complete=False,
        note="Pagination cursor did not advance.",
    )
    monkeypatch.setattr(ninja_agent, "op_by_path", lambda path: {"id": "getEndUsers"})
    monkeypatch.setattr(ninja_agent, "fetch", lambda op, path, query: rows)

    answer = ninja_agent.recipe_latest_end_user("i want last uend user created name")
    assert "cannot safely identify" in answer
    assert "Pagination cursor did not advance" in answer


def test_latest_end_user_recipe_does_not_capture_ordinary_list_request():
    assert ninja_agent.recipe_latest_end_user("show me the list of end users") is None


def test_activity_match_requires_target_and_creation_event():
    target = {"id": 185, "firstName": "Alexander", "lastName": "Dinkel", "email": "alex@example.com"}
    created = {
        "id": 9001,
        "activityType": "SYSTEM",
        "subject": "End user created",
        "message": "Created end user Alexander Dinkel (alex@example.com)",
        "userId": 8,
    }
    unrelated = {
        "id": 9002,
        "activityType": "SYSTEM",
        "subject": "End user updated",
        "message": "Updated end user Alexander Dinkel (alex@example.com)",
        "userId": 8,
    }
    other_user = {
        "id": 9003,
        "activityType": "SYSTEM",
        "subject": "End user created",
        "message": "Created end user Other Person (other@example.com)",
        "userId": 8,
    }
    assert ninja_agent._activity_matches_end_user_creation(created, target)
    assert not ninja_agent._activity_matches_end_user_creation(unrelated, target)
    assert not ninja_agent._activity_matches_end_user_creation(other_user, target)


def test_latest_end_user_creator_is_resolved_from_activity_actor(monkeypatch):
    target = {"id": 185, "firstName": "Alexander", "lastName": "Dinkel", "email": "alex@example.com"}
    operations = {
        "/v2/activities": {"params": [{"name": "pageSize"}, {"name": "olderThan"}, {"name": "class"}]},
        "/v2/users": {"params": []},
    }
    monkeypatch.setattr(ninja_agent, "op_by_path", lambda path: operations.get(path))
    monkeypatch.setattr(ninja_agent, "http_get", lambda path, query: {
        "activities": [{
            "id": 9001,
            "activityType": "SYSTEM",
            "subject": "End user created",
            "message": "Created end user Alexander Dinkel (alex@example.com)",
            "userId": 8,
        }]
    })
    monkeypatch.setattr(ninja_agent, "fetch", lambda op, path, query: ninja_agent.FetchRows([
        {"id": 8, "firstName": "Morgan", "lastName": "Administrator", "email": "morgan@example.com"}
    ]))
    answer = ninja_agent._find_end_user_creator(target)
    assert "Creator verified: Morgan Administrator" in answer
    assert "morgan@example.com" in answer
    assert "user ID 8" in answer


def test_latest_end_user_creator_followup_reuses_previous_latest_user_context(monkeypatch):
    calls = []
    def fake_latest(query, include_creator=False):
        calls.append((query, include_creator))
        return "creator lookup result"
    monkeypatch.setattr(ninja_agent, "recipe_latest_end_user", fake_latest)
    history = [
        {"role": "assistant", "content": "The end user with the highest ID in the complete inventory is Alexander Dinkel."}
    ]
    assert ninja_agent.fast_path("who created?", history) == "creator lookup result"
    assert calls and calls[0][1] is True
