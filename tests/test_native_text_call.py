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

    def test_named_user_count_strips_ninjaone_context(self):
        payload = {"total_matches": 1, "matches": [
            {"id": 2, "firstName": "Richard", "lastName": "Otrebski", "email": "richard.ot@ger.com"}
        ]}
        with patch.object(ninja_agent, "t_resolve", return_value=json.dumps(payload)) as resolve:
            answer = ninja_agent.recipe_count_named("How many users with name Richard in NinjaOne?")
        self.assertEqual(resolve.call_args.args[0]["name"], "Richard")
        self.assertIn("1 NinjaOne user record(s)", answer)

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

    def test_device_prefix_count_uses_complete_inventory_without_llm(self):
        rows = ninja_agent.FetchRows([
            {"systemName": "VM-001"},
            {"systemName": "vm-002"},
            {"systemName": "PC-003"},
            {"name": "VM-004"},
        ])
        with patch.object(ninja_agent, "op_by_path", return_value={"id": "getDevices"}):
            with patch.object(ninja_agent, "fetch", return_value=rows) as fetch:
                answer = ninja_agent.recipe_count_device_prefix(
                    "can i get count of device with name start with VM"
                )
        self.assertEqual(answer, '3 devices have names starting with "VM".')
        fetch.assert_called_once_with({"id": "getDevices"}, "/v2/devices", {})

    def test_device_prefix_count_discloses_partial_inventory(self):
        rows = ninja_agent.FetchRows(
            [{"systemName": "VM-001"}], complete=False, note="Only the first page was returned."
        )
        with patch.object(ninja_agent, "op_by_path", return_value={"id": "getDevices"}):
            with patch.object(ninja_agent, "fetch", return_value=rows):
                answer = ninja_agent.recipe_count_device_prefix(
                    "how many devices start with VM"
                )
        self.assertIn("At least 1", answer)
        self.assertIn("complete total is unknown", answer)

    def test_organization_names_only_request_remains_concise(self):
        rows = [{"id": 6, "name": "HOERBIGER"}]
        with patch.object(ninja_agent, "op_by_path", return_value={"operationId": "getOrganizations"}), patch.object(
            ninja_agent, "fetch", return_value=rows
        ):
            answer = ninja_agent.recipe_list_organizations("List organization names.")
        self.assertIn("1. HOERBIGER", answer)
        self.assertNotIn("ID:", answer)

    def test_fast_count_never_claims_total_from_partial_page(self):
        rows = ninja_agent.FetchRows([{"id": 1}], complete=False, note="cursor unavailable")
        with patch.object(ninja_agent, "op_by_path", return_value={"id": "getDevices"}), patch.object(
            ninja_agent, "fetch", return_value=rows
        ):
            answer = ninja_agent.fast_path("how many devices")
        self.assertIn("At least 1", answer)
        self.assertIn("total is unknown", answer)
        self.assertIn("cursor unavailable", answer)

    def test_named_user_count_marks_partial_matches_as_lower_bound(self):
        payload = {
            "total_matches": 3,
            "matches": [{"firstName": "Rene", "lastName": "Example"}],
            "partial": True,
            "pagination_note": "Only the first page was returned.",
        }
        with patch.object(ninja_agent, "t_resolve", return_value=json.dumps(payload)):
            answer = ninja_agent.recipe_count_named("how many users are there with name rene?")
        self.assertIn("At least 3", answer)
        self.assertIn("tenant-wide total is unknown", answer)
        self.assertIn("Only the first page", answer)

    def test_organization_list_discloses_partial_result(self):
        rows = ninja_agent.FetchRows(
            [{"id": 6, "name": "HOERBIGER"}], complete=False, note="pagination not documented"
        )
        with patch.object(ninja_agent, "op_by_path", return_value={"operationId": "getOrganizations"}), patch.object(
            ninja_agent, "fetch", return_value=rows
        ):
            answer = ninja_agent.recipe_list_organizations("List organization names.")
        self.assertIn("partial data only", answer.lower())
        self.assertIn("HOERBIGER", answer)


if __name__ == "__main__":
    unittest.main()
