"""Regression tests for readable Native API summaries."""

import ninja_agent


def test_record_summary_groups_identity_status_and_organization():
    result = ninja_agent.format_record_summary({
        "systemName": "WS-001",
        "id": 42,
        "status": "ONLINE",
        "lastContact": "2026-11-18T08:00:00Z",
        "organizationId": 6,
        "organizationName": "HOERBIGER",
    }, title="System summary")

    assert "**Identity**" in result
    assert "system Name: WS-001" in result
    assert "**Status and activity**" in result
    assert "status: ONLINE" in result
    assert "**Organization and management**" in result
    assert "organization Name: HOERBIGER" in result


def test_record_summary_formats_lists_and_nested_values_without_raw_json():
    result = ninja_agent.format_record_summary({
        "name": "WS-002",
        "roles": ["Admin", "Support"],
        "location": {"name": "Vienna", "id": 3},
        "enabled": True,
    })

    assert "roles: Admin, Support" in result
    assert "location: name: Vienna; id: 3" in result
    assert "enabled: Yes" in result
    assert '"roles"' not in result


def test_record_summary_limits_large_scalar_lists_and_reports_omitted_fields():
    result = ninja_agent.format_record_summary({
        "name": "WS-003",
        "tags": [f"tag-{index}" for index in range(12)],
        **{f"customField{index}": f"value-{index}" for index in range(45)},
    }, max_fields=5)

    assert "tag-7" in result
    assert "+4 more" in result
    assert "additional field(s) omitted" in result
