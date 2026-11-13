"""Unit tests for local Foundry endpoint and model discovery helpers."""
import unittest

from foundry_discovery import (
    _models_from_payload,
    _normalize_base_url,
    _urls_from_status,
    is_vision_model,
)


class FoundryDiscoveryTests(unittest.TestCase):
    def test_normalizes_host_to_v1_endpoint(self):
        self.assertEqual(
            _normalize_base_url("http://127.0.0.1:52072"),
            "http://127.0.0.1:52072/v1",
        )

    def test_preserves_v1_path(self):
        self.assertEqual(
            _normalize_base_url("http://localhost:52072/v1/"),
            "http://localhost:52072/v1",
        )

    def test_normalizes_uppercase_v1_models_path(self):
        self.assertEqual(
            _normalize_base_url("http://127.0.0.1:49934/V1/Models"),
            "http://127.0.0.1:49934/V1",
        )

    def test_preserves_uppercase_v1_base_path(self):
        self.assertEqual(
            _normalize_base_url("http://127.0.0.1:49934/V1"),
            "http://127.0.0.1:49934/V1",
        )

    def test_rejects_non_local_endpoint(self):
        self.assertIsNone(_normalize_base_url("https://example.com/v1"))

    def test_extracts_cli_endpoint(self):
        self.assertEqual(
            _urls_from_status("Endpoint: http://127.0.0.1:50123"),
            ["http://127.0.0.1:50123/v1"],
        )

    def test_parses_loaded_model_ids_and_vision(self):
        models = _models_from_payload({
            "data": [
                {"id": "phi-4-mini"},
                {"id": "qwen3-vl-4b-instruct-generic-cpu"},
            ]
        })
        self.assertEqual([item["id"] for item in models], [
            "phi-4-mini",
            "qwen3-vl-4b-instruct-generic-cpu",
        ])
        self.assertFalse(models[0]["vision"])
        self.assertTrue(models[1]["vision"])
        self.assertTrue(is_vision_model(models[1]["id"], {"models": models}))


if __name__ == "__main__":
    unittest.main()
