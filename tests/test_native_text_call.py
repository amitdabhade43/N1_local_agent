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

    def test_people_search_extracts_name_and_excludes_similar_names(self):
        payloads = {
            "user": {"matches": [
                {"id": 2, "firstName": "Richard", "lastName": "Otrebski", "email": "richard.ot@ger.com", "userType": "TECHNICIAN"},
                {"id": 3, "firstName": "Ricardo", "lastName": "Example", "email": "ricardo@ger.com"},
            ], "total_matches": 1},
            "end_user": {"matches": [], "total_matches": 0},
            "technician": {"matches": [
                {"id": 2, "firstName": "Richard", "lastName": "Otrebski", "email": "richard.ot@ger.com", "userType": "TECHNICIAN"}
            ], "total_matches": 1},
            "contact": {"matches": [], "total_matches": 0},
        }
        calls = []
        def resolve(args):
            calls.append(args)
            return json.dumps(payloads[args["kind"]])
        with patch.object(ninja_agent, "t_resolve", side_effect=resolve):
            answer = ninja_agent.recipe_search_people("All users with name Richard in NinjaOne")
        self.assertTrue(all(call["name"] == "Richard" for call in calls))
        self.assertIn("Richard Otrebski", answer)
        self.assertNotIn("Ricardo", answer)
        self.assertIn("richard.ot@ger.com", answer)
        self.assertIn("ID: 2", answer)

    def test_t_resolve_people_uses_exact_name_tokens_not_fuzzy_similarity(self):
        rows = [
            {"id": 1, "firstName": "Ricardo", "lastName": "Example"},
            {"id": 2, "firstName": "Richard", "lastName": "Otrebski"},
            {"id": 3, "firstName": "Shaju", "lastName": "Richardson"},
        ]
        with patch.object(ninja_agent, "op_by_path", return_value={"id": "getUsers"}), patch.object(
            ninja_agent, "fetch", return_value=rows
        ):
            result = json.loads(ninja_agent.t_resolve({"kind": "user", "name": "Richard"}))
        self.assertEqual([row["id"] for row in result["matches"]], [2])
        self.assertEqual(result["total_matches"], 1)

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
