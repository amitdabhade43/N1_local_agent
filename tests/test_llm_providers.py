import os
import unittest
from unittest.mock import Mock, patch

import app


class GatewayProviderTests(unittest.TestCase):
    def test_normalize_full_chat_completions_url(self):
        self.assertEqual(
            app.normalize_gateway_base_url("https://llm-gateway.example.com/v1/chat/completions"),
            "https://llm-gateway.example.com/v1",
        )

    @patch.dict(os.environ, {
        "LLM_GATEWAY_BASE_URL": "https://llm-gateway.example.com/v1",
        "LLM_GATEWAY_API_KEY": "secret-test-token",
        "LLM_GATEWAY_MODEL": "",
    }, clear=False)
    @patch("app.requests.get")
    def test_gateway_model_discovery_never_returns_token(self, get):
        response = Mock()
        response.json.return_value = {"data": [{"id": "qwen3-14b"}, {"id": "llama-3.3-70b"}]}
        response.raise_for_status.return_value = None
        get.return_value = response

        result = app.discover_gateway()

        self.assertTrue(result["connected"])
        self.assertEqual([item["id"] for item in result["models"]], ["qwen3-14b", "llama-3.3-70b"])
        self.assertNotIn("secret-test-token", repr(result))
        get.assert_called_once_with(
            "https://llm-gateway.example.com/v1/models",
            headers={"Authorization": "Bearer secret-test-token"},
            timeout=8,
        )

    @patch.dict(os.environ, {
        "LLM_GATEWAY_BASE_URL": "https://llm-gateway.example.com/v1",
        "LLM_GATEWAY_API_KEY": "secret-test-token",
        "LLM_GATEWAY_MODEL": "qwen3-14b",
    }, clear=False)
    @patch("app.discover_gateway")
    def test_resolve_gateway_provider_returns_selected_runtime(self, discover):
        discover.return_value = {
            "connected": True,
            "base_url": "https://llm-gateway.example.com/v1",
            "models": [{"id": "qwen3-14b"}],
        }

        model, base_url, api_key = app.resolve_model_provider("gateway", "qwen3-14b")

        self.assertEqual(model, "qwen3-14b")
        self.assertEqual(base_url, "https://llm-gateway.example.com/v1")
        self.assertEqual(api_key, "secret-test-token")

    def test_chat_request_defaults_to_foundry_for_older_clients(self):
        request = app.ChatRequest(model="local-model", message="hello")
        self.assertEqual(request.provider, "foundry")


if __name__ == "__main__":
    unittest.main()
