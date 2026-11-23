"""Regression tests for no-LLM routing and cached provider discovery."""
import os
import time
import unittest
from unittest.mock import AsyncMock, patch

import app
import ninja_native
import foundry_discovery


class ProviderDiscoveryCacheTests(unittest.TestCase):
    def setUp(self):
        self.original_cache = dict(app._gateway_discovery_cache)
        app._gateway_discovery_cache.update(at=0.0, value=None)

    def tearDown(self):
        app._gateway_discovery_cache.update(self.original_cache)

    def test_gateway_discovery_is_cached_between_requests(self):
        result = {"connected": True, "base_url": "https://gateway.example/v1",
                  "models": [{"id": "model-a"}], "error": None}
        with patch.object(app, "_discover_gateway_uncached", return_value=result) as discover:
            first = app.discover_gateway()
            second = app.discover_gateway()
        self.assertEqual(first["models"][0]["id"], "model-a")
        self.assertEqual(second["models"][0]["id"], "model-a")
        self.assertEqual(discover.call_count, 1)

    def test_gateway_discovery_force_refresh_bypasses_cache(self):
        result = {"connected": True, "base_url": "https://gateway.example/v1",
                  "models": [{"id": "model-a"}], "error": None}
        with patch.object(app, "_discover_gateway_uncached", return_value=result) as discover:
            app.discover_gateway()
            app.discover_gateway(force=True)
        self.assertEqual(discover.call_count, 2)

    def test_gateway_failures_are_cached_briefly(self):
        result = {"connected": False, "base_url": "https://gateway.example/v1",
                  "models": [], "error": "temporarily unavailable"}
        with patch.object(app, "_discover_gateway_uncached", return_value=result) as discover:
            app.discover_gateway()
            app.discover_gateway()
        self.assertEqual(discover.call_count, 1)


class FoundryDiscoveryCacheTests(unittest.TestCase):
    def setUp(self):
        self.original_cache = dict(foundry_discovery._CACHE)

    def tearDown(self):
        foundry_discovery._CACHE.update(self.original_cache)

    def test_successful_foundry_discovery_is_cached(self):
        value = {"connected": True, "base_url": "http://127.0.0.1:50000/v1",
                 "models": [{"id": "model-a"}], "error": None}
        foundry_discovery._CACHE.update(at=time.monotonic(), value=value)
        with patch.object(foundry_discovery, "_status_output",
                          side_effect=AssertionError("cache should avoid CLI discovery")):
            self.assertEqual(foundry_discovery.discover_foundry(), value)

    def test_foundry_force_refresh_bypasses_cache(self):
        value = {"connected": True, "base_url": "http://127.0.0.1:50000/v1",
                 "models": [{"id": "model-a"}], "error": None}
        foundry_discovery._CACHE.update(at=time.monotonic(), value=value)
        with patch.object(foundry_discovery, "_status_output", return_value="") as status:
            with patch.object(foundry_discovery, "_fetch_models", return_value=None):
                foundry_discovery.discover_foundry(force=True)
        status.assert_called_once()


class ContextualRoutingTests(unittest.TestCase):
    def test_follow_up_references_skip_no_llm_routing(self):
        examples = [
            "How many of those are online?",
            "What about those devices?",
            "How many of them are online?",
            "Show me the same users",
            "How about them?",
            "List the previous devices",
        ]
        for message in examples:
            with self.subTest(message=message):
                self.assertTrue(app.is_contextual_followup(message))

    def test_self_contained_requests_remain_eligible_for_fast_routing(self):
        examples = [
            "How many devices?",
            "How many devices are there?",
            "List all organizations",
            "Find user Rene Piuk",
            "Count devices starting with VM",
        ]
        for message in examples:
            with self.subTest(message=message):
                self.assertFalse(app.is_contextual_followup(message))


class DirectRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_fast_path_runs_without_provider_discovery(self):
        expected = ("42 devices in NinjaOne.", [])
        with patch.dict(os.environ, {"AGENT_BACKEND": "native"}):
            with patch.object(ninja_native, "try_fast_path", new=AsyncMock(return_value=expected)) as fast:
                result = await app.try_direct_backend_request("session-1", "How many devices?")
        self.assertEqual(result, expected)
        fast.assert_awaited_once_with("session-1", "How many devices?")

    async def test_mcp_direct_request_uses_mcp_tool_without_model(self):
        expected = ("There are 4 devices.", [])
        with patch.dict(os.environ, {"AGENT_BACKEND": "mcp"}):
            with patch.object(app, "get_tools", new=AsyncMock(return_value=[object()])):
                with patch.object(app, "detect_direct_tool_request",
                                  return_value=("list_devices", {}, "count", None)):
                    with patch.object(app, "run_direct_mcp_tool",
                                      new=AsyncMock(return_value=expected)) as direct:
                        result = await app.try_direct_backend_request("session-2", "How many devices?")
        self.assertEqual(result, expected)
        direct.assert_awaited_once_with("list_devices", {}, "count", None)

    async def test_mcp_complex_request_falls_through_to_model_route(self):
        with patch.dict(os.environ, {"AGENT_BACKEND": "mcp"}):
            with patch.object(app, "get_tools", new=AsyncMock(return_value=[object()])):
                with patch.object(app, "detect_direct_tool_request", return_value=None):
                    result = await app.try_direct_backend_request("session-3", "Investigate why this device is unhealthy.")
        self.assertIsNone(result)

    async def test_native_preflight_calls_fast_path_without_model_client(self):
        with patch.object(ninja_native, "_ensure"):
            with patch.object(ninja_native.core, "set_active_session"):
                with patch.object(ninja_native.core, "clear_active_session"):
                    with patch.object(ninja_native.core, "fast_path", return_value="4 devices") as fast:
                        result = await ninja_native.try_fast_path("session-4", "How many devices?")
        self.assertEqual(result[0], "4 devices")
        self.assertEqual(fast.call_args.args[0], "How many devices?")
        self.assertEqual(fast.call_args.args[1], [])

    async def test_native_fast_path_timeout_falls_through_to_model_route(self):
        import requests
        with patch.object(ninja_native, "_ensure"):
            with patch.object(ninja_native.core, "set_active_session"):
                with patch.object(ninja_native.core, "clear_active_session"):
                    with patch.object(ninja_native.core, "fast_path", side_effect=requests.Timeout()):
                        result = await ninja_native.try_fast_path("session-5", "How many devices?")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
