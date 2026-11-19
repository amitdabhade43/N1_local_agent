import ninja_agent
import ninja_openapi


def sample_spec():
    return {
        "openapi": "3.0.1",
        "info": {"title": "test", "version": "1"},
        "components": {
            "schemas": {
                "EndUser": {
                    "type": "object",
                    "required": ["firstName", "email"],
                    "properties": {
                        "firstName": {"type": "string", "description": "User first name"},
                        "lastName": {"type": "string"},
                        "email": {"type": "string", "format": "email", "description": "Email address"},
                        "profile": {
                            "type": "object",
                            "required": ["enabled"],
                            "properties": {
                                "enabled": {"type": "boolean", "description": "Portal enabled"},
                                "level": {"type": "string", "enum": ["basic", "full"]},
                            },
                        },
                        "devices": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "deviceId": {"type": "integer"},
                                    "deviceName": {"type": "string"},
                                },
                            },
                        },
                    },
                }
            }
        },
        "paths": {
            "/v2/user/end-users": {
                "post": {
                    "operationId": "createEndUser",
                    "summary": "Create an end user",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/EndUser"}
                            }
                        },
                    },
                    "responses": {
                        "201": {
                            "description": "Created",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/EndUser"}
                                }
                            },
                        }
                    },
                },
                "get": {
                    "operationId": "getEndUsers",
                    "summary": "List end users",
                    "parameters": [
                        {
                            "name": "pageSize",
                            "in": "query",
                            "required": False,
                            "description": "Page size",
                            "schema": {"type": "integer", "minimum": 1},
                        }
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
            }
        },
    }


def test_field_catalog_contains_nested_array_and_required_fields():
    spec = sample_spec()
    operations = ninja_openapi.build_operations(spec)
    fields = ninja_openapi.build_field_catalog(spec, operations)

    first_name = next(item for item in fields if item.get("schema_name") == "EndUser"
                      and item.get("path") == "EndUser.firstName")
    email = next(item for item in fields if item.get("schema_name") == "EndUser"
                 and item.get("path") == "EndUser.email")
    enabled = next(item for item in fields if item.get("schema_name") == "EndUser"
                   and item.get("path") == "EndUser.profile.enabled")
    device_id = next(item for item in fields if item.get("schema_name") == "EndUser"
                     and item.get("path") == "EndUser.devices[].deviceId")

    assert first_name["required"] is True
    assert email["format"] == "email"
    assert email["required"] is True
    assert enabled["required"] is True
    assert device_id["type"] == "integer"


def test_inspect_api_returns_all_operation_fields_without_depth_or_count_truncation():
    spec = sample_spec()
    operations = ninja_openapi.build_operations(spec)
    ninja_agent.OPS = operations

    result = __import__("json").loads(ninja_agent.t_inspect({"operation_id": "createEndUser"}))
    paths = {field["path"] for field in result["schema_fields"]}

    assert "body.firstName" in paths
    assert "body.profile.enabled" in paths
    assert "body.devices[].deviceId" in paths
    assert "response[201].email" in paths
    assert result["schema_fields_complete"] is True
    assert result["schema_fields_total"] == len(result["schema_fields"])


def test_search_fields_finds_component_and_operation_fields():
    spec = sample_spec()
    operations = ninja_openapi.build_operations(spec)
    ninja_agent.OPS = operations
    ninja_agent.FIELD_INDEX = ninja_openapi.build_field_catalog(spec, operations)

    result = __import__("json").loads(ninja_agent.t_search_fields({"query": "email address"}))
    assert result["total_matches"] > 0
    assert any(item["name"] == "email" for item in result["matches"])


def test_inspect_api_paginates_fields_without_losing_remaining_fields():
    spec = sample_spec()
    ninja_agent.OPS = ninja_openapi.build_operations(spec)

    first = __import__("json").loads(ninja_agent.t_inspect({
        "operation_id": "createEndUser", "limit": 2, "offset": 0
    }))
    collected = list(first["schema_fields"])
    offset = first["schema_fields_next_offset"]

    assert first["schema_fields_complete"] is False
    while offset is not None:
        page = __import__("json").loads(ninja_agent.t_inspect({
            "operation_id": "createEndUser", "limit": 2, "offset": offset
        }))
        assert page["schema_fields_offset"] == offset
        collected.extend(page["schema_fields"])
        offset = page["schema_fields_next_offset"]

    assert len(collected) == first["schema_fields_total"]
    assert len({(field["source"], field["path"], field.get("location")) for field in collected}) == len(collected)


def test_slim_preserves_nested_fields_and_redacts_secrets():
    result = ninja_agent.slim({
        "id": 10,
        "profile": {"enabled": False, "level": "full"},
        "devices": [{"deviceId": 42, "deviceName": "WS-42"}],
        "apiToken": "must-not-leak",
    })

    assert result["id"] == 10
    assert result["profile"]["enabled"] is False
    assert result["devices_count"] == 1
    assert result["devices"][0]["deviceId"] == 42
    assert result["apiToken"] == "[redacted]"


def test_shape_supports_nested_field_selection_and_filtering():
    rows = [
        {"id": 1, "profile": {"status": "active", "enabled": True}},
        {"id": 2, "profile": {"status": "disabled", "enabled": False}},
    ]

    result = ninja_agent.shape(
        rows, where={"profile.status": "active"}, fields=["id", "profile.enabled"]
    )

    assert result["total"] == 1
    assert result["rows"] == [{"id": 1, "profile.enabled": True}]



def test_text_call_parser_accepts_search_fields():
    assert ninja_agent.text_call('search_fields(query="email address", limit=5)') == (
        "search_fields", {"query": "email address", "limit": 5}
    )
