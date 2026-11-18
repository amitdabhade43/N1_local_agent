"""Regression tests for deterministic descriptive device summaries."""

import json
from unittest.mock import patch

import ninja_agent


def test_descriptive_device_summary_uses_read_only_hardware_endpoints():
    base_details = "NinjaOne system details for VM-6804B761 (device ID 50)."
    operations = {
        "/v2/device/{id}/processors": {"id": "getDeviceProcessors"},
        "/v2/device/{id}/disks": {"id": "getDeviceDiskDrives"},
        "/v2/device/{id}/network-interfaces": {"id": "getDeviceNetworkInterfaces"},
        "/v2/device/{id}/volumes": {"id": "getDeviceVolumes"},
    }
    responses = {
        "getDeviceProcessors": {"total": 1, "rows": [{"name": "Intel Xeon", "numCores": 1}]},
        "getDeviceDiskDrives": {"total": 1, "rows": [{"model": "QEMU HARDDISK", "status": "OK"}]},
        "getDeviceNetworkInterfaces": {"total": 1, "rows": [{"interfaceName": "Ethernet", "status": "Up"}]},
        "getDeviceVolumes": {"total": 1, "rows": [{"name": "C:", "capacity": 100, "freeSpace": 40}]},
    }

    with patch.object(ninja_agent, "recipe_device_details", return_value=base_details), \
         patch.object(ninja_agent, "op_by_path", side_effect=lambda path: operations.get(path)), \
         patch.object(ninja_agent, "t_call", side_effect=lambda args: json.dumps(responses[args["operation_id"]])) as call:
        answer = ninja_agent.fast_path("Give me a descriptive summary of device VM-6804B761")

    assert "NinjaOne system details for VM-6804B761" in answer
    assert "**Processors**" in answer
    assert "Intel Xeon" in answer
    assert "**Disk drives**" in answer
    assert "QEMU HARDDISK" in answer
    assert "**Network interfaces**" in answer
    assert "Ethernet" in answer
    assert "**Storage volumes**" in answer
    assert "C:" in answer
    assert "approval request" not in answer.lower()
    assert call.call_count == 4
    assert all(call_args.args[0]["path_params"] == {"id": 50} for call_args in call.call_args_list)


def test_descriptive_device_summary_reports_optional_read_failure_without_claiming_write():
    base_details = "NinjaOne system details for VM-6804B761 (device ID 50)."
    operations = {
        "/v2/device/{id}/processors": {"id": "getDeviceProcessors"},
        "/v2/device/{id}/disks": {"id": "getDeviceDiskDrives"},
        "/v2/device/{id}/network-interfaces": {"id": "getDeviceNetworkInterfaces"},
        "/v2/device/{id}/volumes": {"id": "getDeviceVolumes"},
    }

    def call(args):
        if args["operation_id"] == "getDeviceNetworkInterfaces":
            return json.dumps({"error": "Request validation failed", "details": ["bad query"]})
        return json.dumps({"total": 0, "rows": []})

    with patch.object(ninja_agent, "recipe_device_details", return_value=base_details), \
         patch.object(ninja_agent, "op_by_path", side_effect=lambda path: operations.get(path)), \
         patch.object(ninja_agent, "t_call", side_effect=call):
        answer = ninja_agent.recipe_device_summary("Give me a summary of device VM-6804B761")

    assert "Not retrieved: Request validation failed" in answer
    assert "No records returned" in answer
    assert "does not establish a complete health assessment" in answer
    assert "approval request" not in answer.lower()
