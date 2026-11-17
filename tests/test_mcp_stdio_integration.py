import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import app


class MCPStdioConfigurationTests(unittest.TestCase):
    def setUp(self):
        app._mcp_clients.clear()

    def tearDown(self):
        app._mcp_clients.clear()

    def test_stdio_gateway_receives_shared_ninjaone_and_model_environment(self):
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server_dir = root / "NinjaRMM-API-Gateway-MCP"
            python_exe = root / "mcp-python.exe"
            server_dir.mkdir()
            python_exe.write_text("", encoding="utf-8")
            captured = {}

            def fake_client(config):
                captured.update(config)
                return SimpleNamespace(config=config)

            with (
                patch.object(app, "MCP_TRANSPORT", "stdio"),
                patch.object(app, "MultiServerMCPClient", side_effect=fake_client),
                patch.object(app, "discover_foundry", return_value={
                    "base_url": "http://127.0.0.1:64343/v1",
                    "models": [{"id": "qwen3-14b-generic-gpu"}],
                }),
                patch.dict(os.environ, {
                    "NINJA_MCP_SERVER_DIR": str(server_dir),
                    "NINJA_MCP_PYTHON": str(python_exe),
                    "NINJA_CLIENT_ID": "native-client-id",
                    "NINJA_CLIENT_SECRET": "native-client-secret",
                    "NINJA_BASE_URL": "https://eu.ninjarmm.com",
                    "NINJA_SCOPE": "monitoring management",
                    "NINJA_MCP_APPROVAL_SECRET": "test-approval-secret",
                    "LLM_GATEWAY_API_KEY": "must-not-be-forwarded",
                }, clear=False),
            ):
                key, client = app._mcp_connection(
                    "qwen3-14b-generic-gpu",
                    "http://127.0.0.1:64343/v1",
                    "local",
                )

            self.assertTrue(key.startswith("stdio:"))
            self.assertIsNotNone(client)
            connection = captured["ninjaone"]
            self.assertEqual(connection["transport"], "stdio")
            self.assertEqual(connection["command"], str(python_exe.resolve()))
            self.assertEqual(connection["cwd"], str(server_dir.resolve()))
            child_env = connection["env"]
            self.assertEqual(child_env["NINJAONE_CLIENT_ID"], "native-client-id")
            self.assertEqual(child_env["NINJAONE_CLIENT_SECRET"], "native-client-secret")
            self.assertEqual(child_env["NINJAONE_INSTANCE"], "eu.ninjarmm.com")
            self.assertEqual(child_env["NINJAONE_SCOPE"], "monitoring management")
            self.assertEqual(child_env["NINJA_MCP_APPROVAL_SECRET"], "test-approval-secret")
            self.assertEqual(child_env["NINJA_MCP_LLM_BASE_URL"], "http://127.0.0.1:64343/v1")
            self.assertEqual(child_env["NINJA_MCP_LLM_MODEL"], "qwen3-14b-generic-gpu")
            self.assertEqual(child_env["NINJA_MCP_LLM_API_KEY"], "local")
            self.assertNotIn("LLM_GATEWAY_API_KEY", child_env)
            self.assertNotIn("FOUNDRY_API_KEY", child_env)

    def test_management_scope_is_required_for_mcp_writes(self):
        from unittest.mock import patch

        with patch.dict(os.environ, {"NINJA_SCOPE": "monitoring"}, clear=False):
            self.assertIn("does not include 'management'", app._mcp_scope_error())

        with patch.dict(os.environ, {"NINJA_SCOPE": "monitoring management"}, clear=False):
            self.assertIsNone(app._mcp_scope_error())

    def test_mcp_approval_token_uses_gateway_canonical_body_contract(self):
        import base64
        import hashlib
        import hmac
        import json
        from unittest.mock import patch

        body = {"addDeviceIds": [24, 32], "removeDeviceIds": []}
        with patch.dict(os.environ, {"NINJA_MCP_APPROVAL_SECRET": "shared-test-secret"}, clear=False):
            token = app._mint_mcp_approval_token("PATCH", "/v2/user/end-user/42/device-access", body)

        encoded, signature = token.split(".", 1)
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        expected_digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        expected_signature = hmac.new(
            b"shared-test-secret", encoded.encode("ascii"), hashlib.sha256
        ).digest()
        supplied_signature = base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
        self.assertEqual(payload["method"], "PATCH")
        self.assertEqual(payload["path"], "/v2/user/end-user/42/device-access")
        self.assertEqual(payload["body_sha256"], expected_digest)
        self.assertTrue(hmac.compare_digest(expected_signature, supplied_signature))
        self.assertGreater(payload["exp"], 0)
        self.assertTrue(payload["jti"])

    def test_approved_write_uses_cached_stateless_tool_after_agent_session(self):
        import asyncio
        import json
        import time
        from unittest.mock import patch

        class FakeTool:
            name = "execute_endpoint"

            def __init__(self):
                self.calls = []

            async def ainvoke(self, arguments):
                self.calls.append(arguments)
                return json.dumps({"executed": True, "data": {"ok": True}})

        action_id = "unit-test-action"
        client_key = "stdio:unit-test-client"
        fake_tool = FakeTool()
        app._tools_by_config[client_key] = [fake_tool]
        app._mcp_pending_actions[action_id] = {
            "action_id": action_id,
            "session_id": "unit-test-session",
            "tool": None,
            "tool_name": "execute_endpoint",
            "client_key": client_key,
            "arguments": {
                "endpoint": "PATCH /v2/devices/{id}",
                "path_params": {"id": 24},
                "body": {"enabled": False},
            },
            "method": "PATCH",
            "path": "/v2/devices/24",
            "body": {"enabled": False},
            "expires_at": time.time() + 300,
        }

        try:
            with patch.dict(os.environ, {
                "NINJA_SCOPE": "monitoring management",
                "NINJA_MCP_APPROVAL_SECRET": "unit-test-secret",
            }, clear=False):
                result = asyncio.run(app.execute_mcp_pending_action(action_id, "unit-test-session"))

            self.assertTrue(result["ok"])
            self.assertEqual(len(fake_tool.calls), 1)
            self.assertIn("approval_token", fake_tool.calls[0])
            self.assertNotIn(action_id, app._mcp_pending_actions)
            repeated = asyncio.run(app.execute_mcp_pending_action(action_id, "unit-test-session"))
            self.assertFalse(repeated["ok"])
            self.assertEqual(len(fake_tool.calls), 1)
        finally:
            app._mcp_pending_actions.pop(action_id, None)
            app._tools_by_config.pop(client_key, None)

    def test_missing_management_scope_does_not_execute_or_consume_pending_write(self):
        import asyncio
        import time
        from unittest.mock import patch

        class FakeTool:
            name = "execute_endpoint"

            def __init__(self):
                self.calls = 0

            async def ainvoke(self, arguments):
                self.calls += 1
                return '{"executed": true}'

        action_id = "scope-blocked-action"
        client_key = "stdio:scope-test-client"
        fake_tool = FakeTool()
        app._tools_by_config[client_key] = [fake_tool]
        app._mcp_pending_actions[action_id] = {
            "action_id": action_id,
            "session_id": "scope-test-session",
            "tool": None,
            "tool_name": "execute_endpoint",
            "client_key": client_key,
            "arguments": {"endpoint": "PATCH /v2/devices/{id}", "path_params": {"id": 24}},
            "method": "PATCH",
            "path": "/v2/devices/24",
            "body": None,
            "expires_at": time.time() + 300,
        }

        try:
            with patch.dict(os.environ, {"NINJA_SCOPE": "monitoring"}, clear=False):
                result = asyncio.run(app.execute_mcp_pending_action(action_id, "scope-test-session"))
            self.assertFalse(result["ok"])
            self.assertIn("management", result["error"])
            self.assertEqual(fake_tool.calls, 0)
            self.assertIn(action_id, app._mcp_pending_actions)
        finally:
            app._mcp_pending_actions.pop(action_id, None)
            app._tools_by_config.pop(client_key, None)

    def test_mcp_write_request_resolves_endpoint_path_parameters(self):
        result = app._mcp_write_request(
            "execute_endpoint",
            {
                "endpoint": "PATCH /v2/user/end-user/{id}/device-access",
                "path_params": {"id": 42},
                "body": {"addDeviceIds": [24]},
            },
        )
        self.assertEqual(
            result,
            ("PATCH", "/v2/user/end-user/42/device-access", {"addDeviceIds": [24]}),
        )

    def test_mcp_delete_is_recognized_for_blocking(self):
        result = app._mcp_write_request(
            "execute_raw_request",
            {"request": "DELETE /v2/device/42"},
        )
        self.assertEqual(result, ("DELETE", "/v2/device/42", None))


if __name__ == "__main__":
    unittest.main()
