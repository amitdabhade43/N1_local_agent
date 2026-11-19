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
    ninja_agent.FIELD_SEARCH_INDEX = ninja_agent._prepare_field_search_index(ninja_agent.FIELD_INDEX)

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



def test_supplemental_yaml_schema_fields_are_indexed_without_replacing_json_operations():
    primary = sample_spec()
    supplemental = {
        "openapi": "3.0.1",
        "info": {"version": "1"},
        "components": {"schemas": {
            "Technician": {
                "type": "object",
                "properties": {
                    "administrator": {"type": "boolean", "description": "System administrator flag"},
                    "roles": {"type": "array", "items": {"type": "string"}},
                },
            },
            "EndUser": {
                "type": "object",
                "properties": {"email": {"type": "string", "format": "email"}},
            },
        }},
        "paths": primary["paths"],
    }
    operations = ninja_openapi.build_operations(primary)
    fields = ninja_openapi.build_field_catalog(
        primary, operations, supplemental_specs=[("yaml", supplemental)]
    )

    admin = next(
        item for item in fields
        if item.get("schema_name") == "Technician"
        and item.get("path") == "Technician.administrator"
    )
    supplemental_email = next(
        item for item in fields
        if item.get("schema_name") == "EndUser"
        and item.get("path") == "EndUser.email"
        and item.get("schema_source") == "yaml"
    )

    assert admin["source"] == "supplemental_component_schema"
    assert admin["type"] == "boolean"
    assert admin["schema_source"] == "yaml"
    assert supplemental_email["format"] == "email"
    search_index = ninja_agent._prepare_field_search_index(fields)
    ninja_agent.FIELD_INDEX = fields
    ninja_agent.FIELD_SEARCH_INDEX = search_index
    matches = __import__("json").loads(ninja_agent.t_search_fields({"query": "administrator"}))
    assert any(item.get("schema_source") == "yaml" for item in matches["matches"])
    assert set(operations) == {"createEndUser", "getEndUsers"}


def test_compare_spec_documents_reports_schema_and_operation_differences():
    primary = sample_spec()
    supplemental = {
        **primary,
        "components": {"schemas": {
            **primary["components"]["schemas"],
            "Technician": {"type": "object", "properties": {"administrator": {"type": "boolean"}}},
        }},
        "paths": {
            **primary["paths"],
            "/v2/extra": {"get": {"operationId": "getExtra", "summary": "Extra"}},
        },
    }

    report = ninja_openapi.compare_spec_documents(primary, supplemental)

    assert report["matching_operation_ids"] == 2
    assert report["supplemental_only_operation_ids"] == ["getExtra"]
    assert report["supplemental_only_schemas"] == ["Technician"]


def test_inspect_schema_reads_yaml_component_schema_and_paginates():
    ninja_agent.SCHEMA_INDEX = {
        ("yaml", "Technician"): {
            "type": "object",
            "properties": {
                "administrator": {"type": "boolean"},
                "roles": {"type": "array", "items": {"type": "string"}},
            },
        }
    }

    first = __import__("json").loads(ninja_agent.t_inspect_schema({
        "schema_name": "Technician", "schema_source": "yaml", "limit": 1, "offset": 0
    }))
    second = __import__("json").loads(ninja_agent.t_inspect_schema({
        "schema_name": "Technician", "schema_source": "yaml", "limit": 1,
        "offset": first["schema_fields_next_offset"],
    }))

    assert first["schema_source"] == "yaml"
    assert first["schema_fields"][0]["path"] == "Technician.administrator"
    assert first["schema_fields_complete"] is False
    assert second["schema_fields_complete"] is True
    assert second["schema_fields"][0]["path"] == "Technician.roles"


def test_text_call_parser_accepts_inspect_schema():
    assert ninja_agent.text_call('inspect_schema(schema_name="Technician", schema_source="yaml")') == (
        "inspect_schema", {"schema_name": "Technician", "schema_source": "yaml"}
    )
