import json
import unittest
from unittest.mock import patch

import ninja_agent


class NativeTextCallTests(unittest.TestCase):
    def test_python_keyword_tool_call(self):
        self.assertEqual(
            ninja_agent.text_call('resolve_entity(kind="user", name="rene")'),
            ("resolve_entity", {"kind": "user", "name": "rene"}),
        )

    def test_json_argument_tool_call(self):
        self.assertEqual(
            ninja_agent.text_call('resolve_entity({"kind": "user", "name": "rene"})'),
            ("resolve_entity", {"kind": "user", "name": "rene"}),
        )

    def test_rejects_non_literal_python_expression(self):
        self.assertIsNone(
            ninja_agent.text_call('resolve_entity(kind=__import__("os"), name="rene")')
        )

    def test_named_user_count_uses_total_matches_not_preview_length(self):
        payload = {
            "total_matches": 7,
            "matches": [
                {"firstName": "Rene", "lastName": "Example", "email": "rene@example.com"}
            ],
        }
        with patch.object(ninja_agent, "t_resolve", return_value=json.dumps(payload)):
            answer = ninja_agent.recipe_count_named("how many users are there with name rene?")
        self.assertIn("7 NinjaOne user record(s)", answer)
        self.assertIn("rene@example.com", answer)

    def test_organization_list_includes_ids_when_requested(self):
        rows = [
            {"id": 6, "name": "HOERBIGER"},
            {"id": 7, "name": "HOERBIGER - Unattended"},
        ]
        with patch.object(ninja_agent, "op_by_path", return_value={"operationId": "getOrganizations"}), patch.object(
            ninja_agent, "fetch", return_value=rows
        ):
            answer = ninja_agent.recipe_list_organizations(
                "List all organizations and their IDs."
            )
        self.assertIn("HOERBIGER — ID: 6", answer)
        self.assertIn("HOERBIGER - Unattended — ID: 7", answer)

    def test_organization_names_only_request_remains_concise(self):
        rows = [{"id": 6, "name": "HOERBIGER"}]
        with patch.object(ninja_agent, "op_by_path", return_value={"operationId": "getOrganizations"}), patch.object(
            ninja_agent, "fetch", return_value=rows
        ):
            answer = ninja_agent.recipe_list_organizations("List organization names.")
        self.assertIn("1. HOERBIGER", answer)
        self.assertNotIn("ID:", answer)


if __name__ == "__main__":
    unittest.main()
