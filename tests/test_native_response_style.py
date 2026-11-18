"""Regression tests for Native response-style guidance."""

import ninja_agent


def test_native_prompt_requests_descriptive_summaries_when_user_asks():
    prompt = ninja_agent.SYSTEM
    assert "summary, overview, analysis, explanation, or report" in prompt
    assert "short direct conclusion" in prompt
    assert "meaningful groups" in prompt
    assert "observed facts from recommendations" in prompt


def test_native_prompt_requires_scope_and_completeness_in_summaries():
    prompt = ninja_agent.SYSTEM
    assert "filters applied" in prompt
    assert "pagination is complete" in prompt
    assert "partial or truncated results" in prompt


def test_native_prompt_prevents_unsupported_interpretations():
    prompt = ninja_agent.SYSTEM
    assert "Never add business impact, root cause, risk level" in prompt
    assert "say what cannot be established" in prompt
