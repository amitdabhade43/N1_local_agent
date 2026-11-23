import uuid

import pytest

from app import _MCPStreamCallbacks


@pytest.mark.asyncio
async def test_mcp_callback_streams_tool_start_and_result_events():
    events = []
    callbacks = _MCPStreamCallbacks(events.append)
    run_id = uuid.uuid4()

    await callbacks.on_tool_start(
        {"name": "ninjaone__call"},
        '{"endpoint":"GET /v2/devices"}',
        run_id=run_id,
        inputs={"endpoint": "GET /v2/devices"},
    )
    await callbacks.on_tool_end(
        '{"endpoint":"GET /v2/devices","result":[]}',
        run_id=run_id,
    )

    activity = [item["activity_event"] for item in events]
    assert activity[0]["kind"] == "call"
    assert activity[0]["tool"] == "ninjaone__call"
    assert activity[1]["kind"] == "result"
    assert activity[1]["success"] is True
