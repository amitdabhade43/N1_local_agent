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
    assert "3 records" in answer
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
