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

    def test_management_scope_is_required_for_mcp_writes(self):
        from unittest.mock import patch

        with patch.dict(os.environ, {"NINJA_SCOPE": "monitoring"}, clear=False):
            self.assertIn("does not include 'management'", app._mcp_scope_error())

        with patch.dict(os.environ, {"NINJA_SCOPE": "monitoring management"}, clear=False):
            self.assertIsNone(app._mcp_scope_error())

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
