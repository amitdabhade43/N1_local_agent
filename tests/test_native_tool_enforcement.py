from types import SimpleNamespace

import ninja_native


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=outcome)])


def fake_client(outcomes):
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_live_inventory_requests_are_marked_as_tool_required():
    assert ninja_native._requires_ninjaone_tool("How many organizations do we have?")
    assert ninja_native._requires_ninjaone_tool("Show device details for VM-123")
    assert ninja_native._requires_ninjaone_tool("Create an end user")
    assert not ninja_native._requires_ninjaone_tool("Hello there")


def test_model_completion_sets_required_tool_choice_when_requested():
    expected = object()
    client, completions = fake_client([expected])

    result = ninja_native._model_completion(client, "local-model", [], require_tool=True)

    assert result is expected
    assert completions.calls[0]["tool_choice"] == "required"
    assert completions.calls[0]["tools"]


def test_model_completion_falls_back_when_provider_rejects_tool_choice():
    expected = object()
    client, completions = fake_client([
        ValueError("unsupported parameter tool_choice"),
        expected,
    ])

    result = ninja_native._model_completion(client, "local-model", [], require_tool=True)

    assert result is expected
    assert len(completions.calls) == 2
    assert completions.calls[0]["tool_choice"] == "required"
    assert "tool_choice" not in completions.calls[1]



def test_large_truncated_live_api_result_can_still_be_marked_successful():
    payload = '{"total": 400, "rows": [' + (" " * 4000) + ']}'
    assert ninja_native._native_tool_result_succeeded("call_api", payload) is True
    assert ninja_native._native_tool_result_succeeded("search_api", payload) is None


def test_api_error_or_approval_payload_is_not_a_verified_read():
    assert ninja_native._native_tool_result_succeeded("call_api", '{"error":"denied"}') is False
    assert ninja_native._native_tool_result_succeeded(
        "call_api", '{"approval_required":true,"action_id":"x"}'
    ) is False



def test_live_entity_resolution_counts_as_a_verified_tenant_lookup():
    assert ninja_native._native_tool_result_succeeded(
        "resolve_entity", '{"matches":[{"id":144,"name":"Rene Piuk"}]}'
    ) is True
