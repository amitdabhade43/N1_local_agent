import app


def test_agent_recursion_limit_defaults_to_enough_mcp_tool_cycles(monkeypatch):
    monkeypatch.delenv("AGENT_RECURSION_LIMIT", raising=False)
    assert app._agent_recursion_limit() == 32


def test_agent_recursion_limit_can_be_configured(monkeypatch):
    monkeypatch.setenv("AGENT_RECURSION_LIMIT", "40")
    assert app._agent_recursion_limit() == 40


def test_agent_recursion_limit_is_bounded(monkeypatch):
    monkeypatch.setenv("AGENT_RECURSION_LIMIT", "500")
    assert app._agent_recursion_limit() == 64
    monkeypatch.setenv("AGENT_RECURSION_LIMIT", "2")
    assert app._agent_recursion_limit() == 8


def test_invalid_agent_recursion_limit_uses_safe_default(monkeypatch):
    monkeypatch.setenv("AGENT_RECURSION_LIMIT", "not-a-number")
    assert app._agent_recursion_limit() == 32
