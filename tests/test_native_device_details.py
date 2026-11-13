import json
import unittest
from unittest.mock import patch

import ninja_agent


class NativeDeviceDetailsTests(unittest.TestCase):
    def test_system_details_resolves_system_name_then_calls_numeric_device_endpoint(self):
        device = {
            "id": 2841,
            "systemName": "2QN9BY3",
            "displayName": "2QN9BY3",
            "serialNumber": "SERIAL-123",
            "match": 1.0,
        }
        detail = {
            "total": 1,
            "rows": [{
                "id": 2841,
                "systemName": "2QN9BY3",
                "nodeClass": "WINDOWS_WORKSTATION",
                "offline": False,
                "ipAddresses": ["10.0.0.25"],
                "references": {"organization": {"id": 6, "name": "Example Org"}},
            }],
        }
        operation = {
            "id": "getDevice",
            "responses": {
                "200": {
                    "content": {
                        "application/json": {
                            "schema": {"properties": {"id": {}, "systemName": {}, "nodeClass": {}}}
                        }
                    }
                }
            },
        }
        with patch.object(
            ninja_agent, "t_resolve",
            return_value=json.dumps({"matches": [device]}),
        ) as resolve, patch.object(
            ninja_agent, "op_by_path", return_value=operation
        ) as op_lookup, patch.object(
            ninja_agent, "t_call", return_value=json.dumps(detail)
        ) as call:
            answer = ninja_agent.fast_path("i want system details of 2QN9BY3")

        resolve.assert_called_once_with({"kind": "device", "name": "2qn9by3"})
        op_lookup.assert_called_once_with("/v2/device/{id}")
        call_args = call.call_args.args[0]
        self.assertEqual(call_args["operation_id"], "getDevice")
        self.assertEqual(call_args["path_params"], {"id": 2841})
        self.assertIn("NinjaOne system details for 2QN9BY3", answer)
        self.assertIn('"nodeClass": "WINDOWS_WORKSTATION"', answer)
        self.assertIn('"10.0.0.25"', answer)

    def test_ambiguous_exact_device_match_requests_clarification(self):
        payload = {
            "matches": [
                {"id": 10, "systemName": "2QN9BY3", "match": 1.0},
                {"id": 11, "displayName": "2QN9BY3", "match": 1.0},
            ]
        }
        with patch.object(ninja_agent, "t_resolve", return_value=json.dumps(payload)):
            answer = ninja_agent.recipe_device_details("system details of 2QN9BY3")
        self.assertIn("Multiple NinjaOne devices exactly match", answer)
        self.assertIn("Please confirm", answer)

    def test_closest_non_exact_match_is_not_used_for_device_details(self):
        payload = {
            "matches": [
                {"id": 10, "systemName": "2QN9BY3-OLD", "match": 0.91},
            ]
        }
        with patch.object(ninja_agent, "t_resolve", return_value=json.dumps(payload)), patch.object(
            ninja_agent, "op_by_path"
        ) as op_lookup:
            answer = ninja_agent.recipe_device_details("system details of 2QN9BY3")
        self.assertIn("could not verify an exact device match", answer)
        op_lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
