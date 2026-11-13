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


if __name__ == "__main__":
    unittest.main()
