import json

import ninja_agent
import ninja_native


def test_shorthand_device_access_request_lists_ambiguous_rene_records(monkeypatch):
    records = {
        "end_user": [
            {"id": 144, "firstName": "Rene", "lastName": "Piuk", "email": "ing.rene.piuk@gmail.com"},
            {"id": 31, "firstName": "PC Admin", "lastName": "Rene Piuk", "email": "admin_pr@hoerbiger.com"},
        ],
        "user": [
            {"id": 144, "firstName": "Rene", "lastName": "Piuk", "email": "ing.rene.piuk@gmail.com", "userType": "END_USER"},
        ],
        "technician": [
            {"id": 2, "firstName": "Rene", "lastName": "Piuk", "email": "rene.piuk@hoerbiger.com", "userType": "TECHNICIAN"},
        ],
        "contact": [],
    }
    calls = []

    def fake_resolve(arguments):
        kind = arguments["kind"]
        calls.append(kind)
        return json.dumps({"matches": records.get(kind, [])})

    def unexpected_write(_arguments):
        raise AssertionError("A shorthand request must not prepare a write while the person is ambiguous.")

    monkeypatch.setattr(ninja_agent, "t_resolve", fake_resolve)
    monkeypatch.setattr(ninja_agent, "t_call", unexpected_write)

    result = ninja_agent.recipe_add_device_access(
        "add VM-6804B761 to rene access"
    )

    assert "Multiple NinjaOne people records matched \"rene\"" in result
    assert "Which user should receive the device access" in result
    assert "Rene Piuk <ing.rene.piuk@gmail.com> (end-user ID 144)" in result
    assert "PC Admin Rene Piuk <admin_pr@hoerbiger.com> (end-user ID 31)" in result
    assert "Rene Piuk <rene.piuk@hoerbiger.com> (technician ID 2)" in result
    assert "console user ID 144" in result
    assert "No change has been prepared" in result
    assert calls == ["end_user", "user", "technician", "contact"]


def test_explicit_device_access_wording_still_matches(monkeypatch):
    calls = []

    def fake_resolve(arguments):
        calls.append(arguments["kind"])
        return json.dumps({"matches": [
            {"id": 31, "firstName": "PC Admin", "lastName": "Rene Piuk",
             "email": "admin_pr@hoerbiger.com"}
        ]})

    monkeypatch.setattr(ninja_agent, "t_resolve", fake_resolve)
    monkeypatch.setattr(
        ninja_agent,
        "t_call",
        lambda _arguments: (_ for _ in ()).throw(
            AssertionError("A single end-user match should continue to preparation; this fixture should be ambiguous.")
        ),
    )

    result = ninja_agent.recipe_add_device_access(
        "add VM-6804B761 to Rene for device access"
    )

    # The first resolver result is the only end-user identity. The recipe proceeds
    # to exact device lookup, which is deliberately not reached in this fixture.
    assert "Could not resolve the device" in result or "No exact device match" in result


def test_shorthand_access_clarification_selection_recovers_original_device():
    history = [
        {"role": "user", "content": "add VM-6804B761 to rene access"},
        {
            "role": "assistant",
            "content": (
                'Multiple NinjaOne people records matched "rene". '
                'Which user should receive the device access for "VM-6804B761"?\n'
                "1. Rene Piuk <ing.rene.piuk@gmail.com> (end-user ID 144)\n"
                "2. PC Admin Rene Piuk <admin_pr@hoerbiger.com> (end-user ID 31)\n"
                "3. Rene Piuk <rene.piuk@hoerbiger.com> (technician ID 2)\n"
                "4. Rene Piuk <ing.rene.piuk@gmail.com> (console user ID 144)\n"
                "Reply with the option number, full name, or email address. No change has been prepared."
            ),
        },
    ]

    resumed, clarification = ninja_native._resolve_device_access_clarification("2", history)

    assert clarification is None
    assert resumed == (
        "add VM-6804B761 to PC Admin Rene Piuk "
        "<admin_pr@hoerbiger.com> for device access"
    )
