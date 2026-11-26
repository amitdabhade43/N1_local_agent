import warnings

import pytest

from ninja_openapi import validate_payload

SPEC = {"components": {"schemas": {
    "Device": {"type": "object", "required": ["organizationId"], "properties": {
        "organizationId": {"type": "integer"},
        "nickname": {"type": "string", "nullable": True},
        "location": {"$ref": "#/components/schemas/Location"},
    }},
    "Location": {"type": "object", "properties": {"floor": {"type": "integer"}}},
}}}
BODY = {"$ref": "#/components/schemas/Device"}


@pytest.fixture(autouse=True)
def no_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        yield


def test_valid_payload_with_local_refs_passes():
    assert validate_payload(BODY, {"organizationId": 7, "location": {"floor": 2}}, SPEC) == []


def test_top_level_ref_violation_is_reported():
    errors = validate_payload(BODY, {"organizationId": "seven"}, SPEC)
    assert errors and errors[0].startswith("organizationId:")


def test_nested_ref_violation_is_reported_with_path():
    errors = validate_payload(BODY, {"organizationId": 7, "location": {"floor": "two"}}, SPEC)
    assert errors and errors[0].startswith("location.floor:")


def test_missing_required_field_is_reported():
    assert any("organizationId" in e for e in validate_payload(BODY, {}, SPEC))


def test_nullable_in_the_body_schema_is_still_normalised():
    inline = {"type": "object", "properties": {"nickname": {"type": "string", "nullable": True}}}
    assert validate_payload(inline, {"nickname": None}, SPEC) == []
    assert validate_payload(inline, {"nickname": 5}, SPEC)


def test_property_named_ref_is_not_mistaken_for_a_reference():
    schema = {"type": "object", "properties": {"$ref": {"type": "string"}}}
    assert validate_payload(schema, {"$ref": "#/not/a/pointer"}, SPEC) == []


def test_unresolvable_ref_returns_message_instead_of_raising():
    errors = validate_payload({"$ref": "#/components/schemas/Missing"}, {"a": 1}, SPEC)
    assert errors and errors[0].startswith("Could not validate request body")


def test_empty_schema_means_nothing_to_validate():
    assert validate_payload(None, {"anything": 1}, SPEC) == []
