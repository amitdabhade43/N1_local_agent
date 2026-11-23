import unittest
from types import SimpleNamespace

from app import detect_direct_tool_request


class MCPDirectRoutingTests(unittest.TestCase):
    def test_technician_firstname_request_routes_to_tool(self):
        tools = [SimpleNamespace(name="list_technicians")]
        self.assertEqual(
            detect_direct_tool_request("list technicians with firstname", tools),
            ("list_technicians", {}, "fields", ["firstName"]),
        )

    def test_user_first_and_last_name_request_routes_to_fields(self):
        tools = [SimpleNamespace(name="list_users")]
        self.assertEqual(
            detect_direct_tool_request("list users with first name and last name", tools),
            ("list_users", {}, "fields", ["firstName", "lastName"]),
        )

    def test_end_users_are_not_misrouted_to_general_users_tool(self):
        tools = [SimpleNamespace(name="list_users")]
        self.assertIsNone(
            detect_direct_tool_request("list end users with first name and last name", tools)
        )

    def test_person_access_question_stays_on_agent_path(self):
        tools = [SimpleNamespace(name="list_devices")]
        self.assertIsNone(
            detect_direct_tool_request("how many devices does Richard have access to?", tools)
        )


    def test_device_deletion_question_uses_shared_fast_path_in_mcp_mode(self):
        import asyncio
        import os
        from unittest.mock import patch
        import app

        async def fake_fast_path(session_id, content):
            return ("shared lifecycle answer", [{"kind": "result", "tool": "fast_path"}])

        async def fake_record(*_args):
            return None

        async def unexpected_get_tools():
            raise AssertionError("MCP model/tool discovery should not run for lifecycle fast path")

        with patch.dict(os.environ, {"AGENT_BACKEND": "mcp"}), \
             patch.object(app, "is_contextual_followup", return_value=False), \
             patch.object(app.ninja_native, "try_fast_path", new=fake_fast_path), \
             patch.object(app, "_record_mcp_direct_turn", new=fake_record), \
             patch.object(app, "get_tools", new=unexpected_get_tools):
            result = asyncio.run(
                app.try_direct_backend_request("test-session", "Who deleted B82ZST3?")
            )

        self.assertEqual(result[0], "shared lifecycle answer")


if __name__ == "__main__":
    unittest.main()
