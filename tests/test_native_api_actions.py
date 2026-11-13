import json
import unittest
from unittest.mock import patch

import ninja_agent as core
from ninja_openapi import build_operations


class NativeApiActionTests(unittest.TestCase):
    def setUp(self):
        self.original = {
            "OPS": core.OPS,
            "IDX": core.IDX,
            "SPEC_DATA": core.SPEC_DATA,
            "SCOPE": core.SCOPE,
        }
        self.spec = {
            "openapi": "3.0.1",
            "paths": {
                "/v2/things": {
                    "post": {
                        "operationId": "createThing",
                        "summary": "Create thing",
                        "requestBody": {
                            "required": True,
                            "content": {"application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["name"],
                                    "properties": {"name": {"type": "string", "minLength": 2}},
                                }
                            }},
                        },
                    },
                    "delete": {
                        "operationId": "deleteThings",
                        "summary": "Delete things",
                    },
                },
                "/v2/things/delete": {
                    "post": {
                        "operationId": "deleteThingsByPost",
                        "summary": "Delete things",
                    }
                },
                "/v2/things/{id}": {
                    "get": {
                        "operationId": "getThing",
                        "summary": "Get thing",
                        "parameters": [
                            {"name": "id", "in": "path", "required": True,
                             "schema": {"type": "integer"}}
                        ],
                    }
                },
                "/v2/ticketing/trigger/board/{boardId}/run": {
                    "post": {
                        "operationId": "getTicketsByBoard",
                        "summary": "List of tickets for board",
                        "parameters": [
                            {"name": "boardId", "in": "path", "required": True,
                             "schema": {"type": "integer"}}
                        ],
                        "requestBody": {
                            "content": {"application/json": {
                                "schema": {"type": "object", "properties": {
                                    "pageSize": {"type": "integer"},
                                    "searchCriteria": {"type": "string"},
                                    "lastCursorId": {"type": "integer"}
                                }}
                            }}
                        },
                        "responses": {
                            "default": {"content": {"application/json": {
                                "schema": {"type": "object", "properties": {
                                    "data": {"type": "array", "items": {"type": "object"}},
                                    "metadata": {"type": "object", "properties": {
                                        "lastCursorId": {"type": "integer"}
                                    }}
                                }}
                            }}}
                        }
                    }
                },
            },
        }
        core.SPEC_DATA = self.spec
        core.OPS = build_operations(self.spec)
        core.IDX = core.Index(core.OPS)
        core.SCOPE = "monitoring management"
        core.set_active_session("session-a")

    def tearDown(self):
        core.clear_active_session()
        core.OPS = self.original["OPS"]
        core.IDX = self.original["IDX"]
        core.SPEC_DATA = self.original["SPEC_DATA"]
        core.SCOPE = self.original["SCOPE"]
        core.clear_pending_session("session-a")

    def test_mutation_is_prepared_and_never_sent_before_approval(self):
        with patch.object(core, "http_request") as request:
            prepared = json.loads(core.t_call({
                "operation_id": "createThing",
                "body": {"name": "Laptop"},
            }))
            self.assertTrue(prepared["approval_required"])
            request.assert_not_called()

            wrong_session = core.execute_pending_action(prepared["action_id"], "session-b")
            self.assertFalse(wrong_session["ok"])
            request.assert_not_called()

            with patch.object(core, "http_request", return_value={"id": 42}) as approved_request:
                result = core.execute_pending_action(prepared["action_id"], "session-a")
                self.assertTrue(result["ok"])
                approved_request.assert_called_once_with(
                    "POST", "/v2/things", {}, {"name": "Laptop"}, "application/json", {},
                    approved=True
                )

            replay = core.execute_pending_action(prepared["action_id"], "session-a")
            self.assertFalse(replay["ok"])

    def test_invalid_body_is_rejected_before_approval(self):
        result = json.loads(core.t_call({
            "operation_id": "createThing",
            "body": {"name": "A"},
        }))
        self.assertEqual(result["error"], "Request validation failed")
        self.assertFalse(result.get("approval_required"))

    def test_tool_definitions_and_dispatch_are_in_sync(self):
        declared = {item["function"]["name"] for item in core.TOOLS}
        self.assertEqual(declared, set(core.DISPATCH))
        self.assertIn("inspect_api", declared)
        self.assertIn("resolve_entity", declared)

    def test_delete_verbs_and_post_delete_actions_are_not_in_catalog(self):
        self.assertNotIn("deleteThings", core.OPS)
        self.assertNotIn("deleteThingsByPost", core.OPS)
        self.assertIn("createThing", core.OPS)

    def test_get_path_parameter_is_validated_and_quoted(self):
        with patch.object(core, "fetch", return_value=[{"id": 7, "name": "Test"}]) as fetch:
            result = json.loads(core.t_call({
                "operation_id": "getThing",
                "path_params": {"id": 7},
            }))
        fetch.assert_called_once()
        self.assertEqual(result["total"], 1)

    def test_documented_read_only_post_query_uses_schema_driven_cursor_pagination(self):
        pages = [
            {"data": [{"id": 5, "subject": "Printer offline"}], "metadata": {"lastCursorId": 5}},
            {"data": [{"id": 6, "subject": "Printer offline - second"}], "metadata": {"lastCursorId": 6}},
            {"data": [], "metadata": {"lastCursorId": 6}},
        ]
        with patch.object(core, "http_request", side_effect=pages) as request:
            result = json.loads(core.t_call({
                "operation_id": "getTicketsByBoard",
                "path_params": {"boardId": 12},
                "body": {"pageSize": 25, "searchCriteria": "printer"},
            }))
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args_list[0].args, (
            "POST", "/v2/ticketing/trigger/board/12/run", {},
            {"pageSize": 25, "searchCriteria": "printer"}, "application/json", {}
        ))
        self.assertEqual(request.call_args_list[1].args[3]["lastCursorId"], 5)
        self.assertEqual(request.call_args_list[2].args[3]["lastCursorId"], 6)
        self.assertTrue(all(call.kwargs == {"read_only": True} for call in request.call_args_list))
        self.assertEqual(result["total"], 2)

    def test_http_boundary_rejects_unapproved_writes_and_all_delete_paths(self):
        with self.assertRaisesRegex(ValueError, "explicit user approval"):
            core.http_request("POST", "/v2/things", body={"name": "Laptop"})
        with self.assertRaisesRegex(ValueError, "Delete execution is disabled"):
            core.http_request("POST", "/v2/things/delete", read_only=True)
        with self.assertRaisesRegex(ValueError, "Delete execution is disabled"):
            core.http_request("DELETE", "/v2/things", approved=True)

    def test_token_uses_current_oauth_route_with_legacy_fallback(self):
        class FakeResponse:
            def __init__(self, status_code, payload=None):
                self.status_code = status_code
                self.payload = payload or {}

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

            def json(self):
                return self.payload

        original = {
            "CID": core.CID,
            "CSECRET": core.CSECRET,
            "BASE": core.BASE,
            "TOKEN_URL": core.TOKEN_URL,
            "token_cache": dict(core._tok),
        }
        try:
            core.CID = "client-id"
            core.CSECRET = "client-secret"
            core.BASE = "https://eu.ninjarmm.com"
            core.TOKEN_URL = ""
            core._tok.update(v=None, exp=0)
            with patch.object(core.requests, "post", side_effect=[
                FakeResponse(404),
                FakeResponse(200, {"access_token": "test-token", "expires_in": 3600}),
            ]) as post:
                self.assertEqual(core.token(), "test-token")
            self.assertEqual(post.call_args_list[0].args[0], "https://eu.ninjarmm.com/oauth/token")
            self.assertEqual(post.call_args_list[1].args[0], "https://eu.ninjarmm.com/ws/oauth/token")
        finally:
            core.CID = original["CID"]
            core.CSECRET = original["CSECRET"]
            core.BASE = original["BASE"]
            core.TOKEN_URL = original["TOKEN_URL"]
            core._tok.update(original["token_cache"])

    def test_write_approval_requires_management_scope(self):
        core.SCOPE = "monitoring"
        prepared = json.loads(core.t_call({
            "operation_id": "createThing",
            "body": {"name": "Laptop"},
        }))
        result = core.execute_pending_action(prepared["action_id"], "session-a")
        self.assertFalse(result["ok"])
        self.assertIn("management", result["error"].lower())


if __name__ == "__main__":
    unittest.main()
