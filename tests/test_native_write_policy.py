import ninja_native


def test_create_end_user_is_a_mutation_and_required_field_question_is_preserved():
    assert ninja_native._is_mutation_request("create end user")
    assert ninja_native._is_action_clarification(
        "To create the end user, please provide first name, last name and email address."
    )


def test_explicit_delete_intent_is_blocked_but_revoke_access_is_not_a_delete():
    assert ninja_native._is_delete_request("delete end user 123")
    assert ninja_native._is_delete_request("permanently remove this user")
    assert ninja_native._is_delete_request("remove end user 123")
    assert not ninja_native._is_delete_request("revoke this end user's device access")
    assert not ninja_native._is_delete_request("remove the user from this role")


def test_delete_request_returns_before_any_model_or_api_call(monkeypatch):
    monkeypatch.setattr(ninja_native, "_ensure", lambda: None)
    monkeypatch.setattr(
        ninja_native,
        "_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model must not run")),
    )
    ninja_native._sessions.pop("delete-policy-test", None)

    answer, activity = ninja_native._run_inner(
        "delete-policy-test", "unused-model", "delete end user 123", "http://127.0.0.1:1"
    )

    assert "can't delete" in answer.lower()
    assert activity[0]["stage"] == "blocked_delete"
    assert ninja_native._sessions["delete-policy-test"][-1]["content"] == answer



def test_create_end_user_without_email_asks_for_required_details(monkeypatch):
    monkeypatch.setattr(
        ninja_native,
        "_ensure",
        lambda: (_ for _ in ()).throw(AssertionError("API catalog must not load yet")),
    )
    ninja_native._sessions.pop("create-end-user-test", None)

    answer, activity = ninja_native._run_inner(
        "create-end-user-test",
        "unused-model",
        "I want to create an end user, can you help me?",
        "http://127.0.0.1:1",
    )

    assert "first name, last name, and email address" in answer.lower()
    assert "nothing will be created until you approve" in answer.lower()
    assert activity[0]["stage"] == "missing_required_fields"
    assert ninja_native._sessions["create-end-user-test"][-1]["content"] == answer



def test_read_only_list_requests_are_not_misclassified_as_mutations():
    assert not ninja_native._is_mutation_request(
        "how many organizations we have and give me list"
    )
    assert not ninja_native._is_mutation_request("give me the list of organizations")
    assert not ninja_native._is_mutation_request("show me users who have device access")


def test_explicit_access_grants_remain_mutations():
    assert ninja_native._is_mutation_request("give Rene access to VM-123")
    assert ninja_native._is_mutation_request("grant device access to Rene")
    assert ninja_native._is_mutation_request("add VM-123 to Rene access")
    assert ninja_native._is_mutation_request("remove device access from Rene")



def test_delete_preflight_blocks_before_loading_the_api_catalog(monkeypatch):
    monkeypatch.setattr(
        ninja_native,
        "_ensure",
        lambda: (_ for _ in ()).throw(AssertionError("blocked delete must not load the API catalog")),
    )
    monkeypatch.setattr(ninja_native.core, "set_active_session", lambda session_id: None)
    monkeypatch.setattr(ninja_native.core, "clear_active_session", lambda: None)

    answer, activity = ninja_native._try_fast_path("delete-before-init", "delete device 123")

    assert "can't delete" in answer.lower()
    assert activity[0]["stage"] == "blocked_delete"
