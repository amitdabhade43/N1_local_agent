import unittest

from ninja_openapi import (
    build_operations,
    describe_operation,
    is_delete_operation,
    validate_payload,
)


class OpenApiCatalogTests(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "openapi": "3.0.1",
            "security": [{"oauth2": ["monitoring", "management"]}],
            "components": {
                "schemas": {
                    "CreateThing": {
                        "type": "object",
                        "required": ["name", "kind"],
                        "properties": {
                            "name": {"type": "string", "minLength": 2},
                            "kind": {"type": "string", "enum": ["device", "user"]},
                            "enabled": {"type": "boolean", "nullable": True},
                        },
                    }
                }
            },
            "paths": {
                "/v2/things": {
                    "parameters": [
                        {"name": "tenant", "in": "query", "required": True,
                         "schema": {"type": "string"}}
                    ],
                    "get": {
                        "operationId": "listThings",
                        "summary": "List things",
                        "parameters": [
                            {"name": "pageSize", "in": "query",
                             "schema": {"type": "integer"}}
                        ],
                    },
                    "post": {
                        "operationId": "createThing",
                        "summary": "Create thing",
                        "requestBody": {
                            "required": True,
                            "content": {"application/json": {
                                "schema": {"$ref": "#/components/schemas/CreateThing"}
                            }},
                        },
                    },
                },
                "/v2/things/{id}": {
                    "patch": {
                        "operationId": "updateThing",
                        "summary": "Update thing",
                        "parameters": [
                            {"name": "id", "in": "path", "required": True,
                             "schema": {"type": "integer"}}
                        ],
                        "requestBody": {
                            "required": True,
                            "content": {"application/json": {
                                "schema": {"type": "object", "required": ["enabled"],
                                           "properties": {"enabled": {"type": "boolean"}}}
                            }},
                        },
                    },
                    "put": {"operationId": "replaceThing", "summary": "Replace thing"},
                    "delete": {"operationId": "deleteThing", "summary": "Delete thing"},
                },
                "/v2/things/delete": {
                    "post": {"operationId": "deleteThings", "summary": "Delete things"},
                },
            },
        }

    def test_catalog_preserves_supported_methods_and_shared_parameters(self):
        operations = build_operations(self.spec)
        self.assertEqual(set(operations), {"listThings", "createThing", "updateThing", "replaceThing"})
        self.assertEqual(operations["listThings"]["method"], "GET")
        self.assertEqual(
            [p["name"] for p in operations["listThings"]["params"]],
            ["tenant", "pageSize"],
        )
        self.assertEqual(operations["createThing"]["request_body"]["media_type"], "application/json")

    def test_request_body_reference_is_expanded_and_required_fields_validated(self):
        operation = build_operations(self.spec)["createThing"]
        schema = operation["request_body"]["schema"]
        self.assertIn("name", schema["properties"])
        errors = validate_payload(schema, {"name": "A", "kind": "invalid"}, self.spec)
        self.assertTrue(any("minLength" in error for error in errors))
        self.assertTrue(any("not one of" in error for error in errors))

    def test_valid_nullable_payload_is_accepted(self):
        operation = build_operations(self.spec)["createThing"]
        errors = validate_payload(
            operation["request_body"]["schema"],
            {"name": "Laptop", "kind": "device", "enabled": None},
            self.spec,
        )
        self.assertEqual(errors, [])

    def test_delete_http_method_and_post_delete_semantics_are_excluded(self):
        self.assertTrue(is_delete_operation("DELETE", "/v2/things/1", "removeThing", "Remove"))
        self.assertTrue(is_delete_operation("POST", "/v2/things/delete", "deleteThings", "Delete things"))
        self.assertFalse(is_delete_operation("POST", "/v2/things", "createThing", "Create thing"))

    def test_operation_description_includes_schema_and_parameter_constraints(self):
        operation = build_operations(self.spec)["createThing"]
        details = describe_operation(operation, include_schema=True)
        self.assertTrue(details["request_body_required"])
        self.assertEqual(details["request_body_schema"]["required"], ["name", "kind"])


if __name__ == "__main__":
    unittest.main()
