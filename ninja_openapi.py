"""OpenAPI-driven NinjaOne operation catalog and request validation.

The catalog is built from the bundled NinjaOne OpenAPI document. It preserves
operation parameters, request bodies, response metadata, and schema references
so the agent does not need endpoint-specific hard-coded handlers.
"""
from __future__ import annotations

import copy
import re
from typing import Any

HTTP_METHODS = ("get", "post", "put", "patch")
DELETE_METHODS = {"DELETE"}


def _pointer(root: dict, ref: str) -> Any:
    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported external OpenAPI reference: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"Unresolved OpenAPI reference: {ref}")
        node = node[part]
    return node


def expand_refs(value: Any, spec: dict, seen: frozenset[str] = frozenset(), depth: int = 0) -> Any:
    """Expand local $refs while retaining a safe marker for recursive schemas."""
    if depth > 24:
        return {"description": "Schema expansion depth limit reached"}
    if isinstance(value, list):
        return [expand_refs(item, spec, seen, depth + 1) for item in value]
    if not isinstance(value, dict):
        return value

    current = value
    if "$ref" in value:
        ref = str(value["$ref"])
        if ref in seen:
            return {"$ref": ref, "description": "Recursive schema reference"}
        target = _pointer(spec, ref)
        expanded = expand_refs(target, spec, seen | {ref}, depth + 1)
        siblings = {k: v for k, v in value.items() if k != "$ref"}
        if isinstance(expanded, dict):
            current = {**expanded, **siblings}
        else:
            current = expanded

    if isinstance(current, dict):
        return {k: expand_refs(v, spec, set(seen), depth + 1) for k, v in current.items()}
    return current


def is_delete_operation(method: str, path: str, operation_id: str, summary: str) -> bool:
    """Block DELETE verbs and POST/PATCH/etc. that are explicitly delete actions."""
    if method.upper() in DELETE_METHODS:
        return True
    haystack = " ".join((path, operation_id, summary)).lower()
    if re.search(r"(^|[/_\s-])delete($|[/_\s-])", haystack):
        return True
    return False


def is_read_only_operation(method: str, operation_id: str, summary: str) -> bool:
    """Recognize the small set of POST operations explicitly documented as reads."""
    method = method.upper()
    if method == "GET":
        return True
    if method != "POST":
        return False
    operation_name = operation_id.lower()
    description = summary.strip().lower()
    read_name = operation_name.startswith(("get", "list", "search", "find", "query", "fetch", "retrieve"))
    read_summary = description.startswith(("get ", "list ", "search ", "find ", "retrieve "))
    return read_name and read_summary


def _media_schema(request_body: dict) -> tuple[str | None, dict | None]:
    content = request_body.get("content") or {}
    # Prefer JSON, then URL-encoded form. Multipart/binary is intentionally
    # described but not silently coerced into JSON.
    for media_type in ("application/json", "application/x-www-form-urlencoded", "multipart/form-data"):
        entry = content.get(media_type)
        if isinstance(entry, dict):
            schema = entry.get("schema")
            return media_type, schema if isinstance(schema, dict) else None
    if content:
        media_type, entry = next(iter(content.items()))
        schema = entry.get("schema") if isinstance(entry, dict) else None
        # NinjaOne uses */* for several structured JSON billing payloads.
        if media_type == "*/*" and isinstance(schema, dict):
            return "application/json", schema
        return media_type, schema if isinstance(schema, dict) else None
    return None, None


def build_operations(spec: dict) -> dict[str, dict]:
    """Build a complete operation map, excluding all delete semantics."""
    operations: dict[str, dict] = {}
    for path, path_item in (spec.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        shared_params = path_item.get("parameters") or []
        for method, raw in path_item.items():
            if method.lower() not in HTTP_METHODS + ("delete",) or not isinstance(raw, dict):
                continue
            method_upper = method.upper()
            operation_id = raw.get("operationId") or f"{method}_{path}"
            summary = raw.get("summary") or ""
            if is_delete_operation(method_upper, path, operation_id, summary):
                continue

            params_by_key = {}
            for raw_param in list(shared_params) + list(raw.get("parameters") or []):
                param = expand_refs(raw_param, spec)
                if not isinstance(param, dict) or not param.get("name"):
                    continue
                key = (param.get("in"), param["name"])
                params_by_key[key] = {
                    "name": param["name"],
                    "in": param.get("in"),
                    "required": bool(param.get("required")),
                    "description": param.get("description", ""),
                    "schema": param.get("schema") or {},
                    "style": param.get("style"),
                    "explode": param.get("explode"),
                }
            params = list(params_by_key.values())

            raw_body = raw.get("requestBody")
            body = expand_refs(raw_body, spec) if raw_body else None
            media_type, body_schema = _media_schema(body) if isinstance(body, dict) else (None, None)
            if body_schema:
                body_schema = expand_refs(body_schema, spec)

            responses = expand_refs(raw.get("responses") or {}, spec)
            operations[operation_id] = {
                "id": operation_id,
                "method": method_upper,
                "read_only": is_read_only_operation(method_upper, operation_id, summary),
                "path": path,
                "summary": summary,
                "description": raw.get("description") or "",
                "tags": raw.get("tags") or [],
                "params": params,
                "request_body": {
                    "required": bool(body.get("required")) if isinstance(body, dict) else False,
                    "description": body.get("description", "") if isinstance(body, dict) else "",
                    "media_type": media_type,
                    "schema": body_schema,
                } if body else None,
                "responses": responses,
                "security": raw.get("security", spec.get("security", [])),
            }
    return operations


def _schema_terms(schema: Any, limit: int = 10000) -> list[str]:
    """Collect field names, descriptions, formats and enum values deeply enough to index the full spec."""
    terms: list[str] = []
    def walk(node: Any, depth: int = 0):
        if len(terms) >= limit or depth > 32:
            return
        if isinstance(node, dict):
            for key in ("title", "description", "format"):
                value = node.get(key)
                if isinstance(value, str):
                    terms.append(value)
            for name, child in (node.get("properties") or {}).items():
                terms.append(str(name))
                walk(child, depth + 1)
            if isinstance(node.get("items"), dict):
                walk(node["items"], depth + 1)
            for key in ("allOf", "oneOf", "anyOf"):
                children = node.get(key)
                if isinstance(children, dict):
                    walk(children, depth + 1)
                elif isinstance(children, list):
                    for child in children:
                        walk(child, depth + 1)
            enum = node.get("enum")
            if isinstance(enum, list):
                terms.extend(str(x) for x in enum[:30])
    walk(schema)
    return terms


def operation_document(operation: dict) -> str:
    body = operation.get("request_body") or {}
    body_schema = body.get("schema") or {}
    param_text = " ".join(
        f'{p.get("name")} {p.get("in")} {p.get("description", "")} {_schema_terms(p.get("schema"))}'
        for p in operation.get("params", [])
    )
    response_terms = []
    for response in (operation.get("responses") or {}).values():
        if not isinstance(response, dict):
            continue
        content = response.get("content") or {}
        json_response = content.get("application/json") or {}
        response_terms.extend(_schema_terms(json_response.get("schema") or {}))
        response_terms.append(response.get("description", ""))
    return " ".join((
        operation.get("id", ""),
        operation.get("method", ""),
        operation.get("path", ""),
        operation.get("summary", ""),
        operation.get("description", ""),
        " ".join(operation.get("tags", [])),
        param_text,
        body.get("description", ""),
        " ".join(_schema_terms(body_schema)),
        " ".join(response_terms),
    ))


def schema_field_entries(
    schema: Any,
    prefix: str = "$",
    *,
    source: str = "",
    required_names: set[str] | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    """Flatten every documented schema property without dropping nested fields."""
    if depth > 32 or not isinstance(schema, dict):
        return []
    required_names = required_names or set()
    entries: list[dict[str, Any]] = []

    def metadata(node: dict, path: str, name: str, required: bool) -> dict[str, Any]:
        item = {
            "name": name,
            "path": path,
            "required": bool(required),
            "type": node.get("type"),
            "format": node.get("format"),
            "description": node.get("description", ""),
            "enum": node.get("enum"),
            "nullable": node.get("nullable"),
            "readOnly": node.get("readOnly"),
            "writeOnly": node.get("writeOnly"),
            "deprecated": node.get("deprecated"),
            "default": node.get("default"),
            "example": node.get("example"),
            "minimum": node.get("minimum"),
            "maximum": node.get("maximum"),
            "minLength": node.get("minLength"),
            "maxLength": node.get("maxLength"),
            "minItems": node.get("minItems"),
            "maxItems": node.get("maxItems"),
            "pattern": node.get("pattern"),
            "multipleOf": node.get("multipleOf"),
            "exclusiveMinimum": node.get("exclusiveMinimum"),
            "exclusiveMaximum": node.get("exclusiveMaximum"),
            "uniqueItems": node.get("uniqueItems"),
            "minProperties": node.get("minProperties"),
            "maxProperties": node.get("maxProperties"),
        }
        return {key: value for key, value in item.items() if value not in (None, "", [], {})}

    properties = schema.get("properties")
    if isinstance(properties, dict):
        parent_required = set(schema.get("required") or [])
        for name, child in properties.items():
            if not isinstance(child, dict):
                continue
            path = f"{prefix}.{name}"
            entries.append(metadata(child, path, str(name), name in parent_required))
            entries.extend(schema_field_entries(
                child, path, source=source,
                required_names=set(child.get("required") or []), depth=depth + 1,
            ))

    items = schema.get("items")
    if isinstance(items, dict):
        entries.extend(schema_field_entries(
            items, prefix + "[]", source=source,
            required_names=set(items.get("required") or []), depth=depth + 1,
        ))

    for composition in ("allOf", "oneOf", "anyOf"):
        children = schema.get(composition)
        if isinstance(children, list):
            for index, child in enumerate(children):
                entries.extend(schema_field_entries(
                    child, f"{prefix}.{composition}[{index}]", source=source,
                    required_names=set((child or {}).get("required") or []) if isinstance(child, dict) else set(),
                    depth=depth + 1,
                ))

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        entries.extend(schema_field_entries(
            additional, prefix + ".*", source=source,
            required_names=set(additional.get("required") or []), depth=depth + 1,
        ))
    return entries


def operation_field_entries(operation: dict) -> list[dict[str, Any]]:
    """Return request, parameter and response fields for one expanded operation."""
    result: list[dict[str, Any]] = []
    operation_id = operation.get("id", "")
    for parameter in operation.get("params", []):
        name = str(parameter.get("name", ""))
        schema = parameter.get("schema") or {}
        entry = {
            "name": name,
            "path": f'{parameter.get("in", "parameter")}.{name}',
            "source": "parameter",
            "location": parameter.get("in"),
            "required": bool(parameter.get("required")),
            "type": schema.get("type"),
            "format": schema.get("format"),
            "description": parameter.get("description", ""),
            "enum": schema.get("enum"),
        }
        result.append({key: value for key, value in entry.items() if value not in (None, "", [], {})})
        result.extend({
            **field,
            "source": "parameter",
            "location": parameter.get("in"),
        } for field in schema_field_entries(schema, f'{parameter.get("in", "parameter")}.{name}'))
    body = operation.get("request_body") or {}
    if body.get("schema"):
        result.extend({
            **field,
            "source": "request",
            "location": "body",
        } for field in schema_field_entries(body["schema"], "body"))
    for status, response in (operation.get("responses") or {}).items():
        if not isinstance(response, dict):
            continue
        content = response.get("content") or {}
        for media_type, media in content.items():
            if not isinstance(media, dict) or not isinstance(media.get("schema"), dict):
                continue
            result.extend({
                **field,
                "source": "response",
                "location": str(status),
                "media_type": media_type,
            } for field in schema_field_entries(media["schema"], f"response[{status}]"))
    for item in result:
        item.setdefault("operation_id", operation_id)
        item.setdefault("method", operation.get("method"))
        item.setdefault("operation_path", operation.get("path"))
    return result


def build_field_catalog(spec: dict, operations: dict[str, dict]) -> list[dict[str, Any]]:
    """Index all component schema fields and all operation-specific parameter/body/response fields."""
    entries: list[dict[str, Any]] = []
    for schema_name, raw_schema in (spec.get("components", {}).get("schemas") or {}).items():
        expanded = expand_refs(raw_schema, spec)
        entries.extend({
            **field,
            "source": "component_schema",
            "schema_name": schema_name,
        } for field in schema_field_entries(expanded, schema_name))
    for operation in operations.values():
        entries.extend(operation_field_entries(operation))
    # Stable de-duplication keeps the catalog compact while retaining distinct sources.
    unique: dict[tuple, dict[str, Any]] = {}
    for entry in entries:
        key = (
            entry.get("source"), entry.get("schema_name"), entry.get("operation_id"),
            entry.get("location"), entry.get("path"), entry.get("name"),
        )
        unique[key] = entry
    return list(unique.values())


def normalize_nullable(schema: Any) -> Any:
    """Convert OpenAPI 3.0 nullable markers into JSON Schema-compatible types."""
    if isinstance(schema, list):
        return [normalize_nullable(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {k: normalize_nullable(v) for k, v in schema.items() if k != "nullable"}
    if schema.get("nullable") is True and isinstance(result.get("type"), str):
        result["type"] = [result["type"], "null"]
    return result


def validate_payload(schema: dict | None, payload: Any, spec: dict) -> list[str]:
    """Validate a request body against the OpenAPI schema, including local refs."""
    if not schema:
        return []
    try:
        from jsonschema import Draft7Validator, RefResolver
    except ImportError:
        return ["Request-body validation requires the jsonschema package. Install requirements.txt."]

    candidate = normalize_nullable(copy.deepcopy(schema))
    try:
        resolver = RefResolver.from_schema(spec)
        validator = Draft7Validator(candidate, resolver=resolver)
        errors = sorted(validator.iter_errors(payload), key=lambda e: list(map(str, e.absolute_path)))
        return [
            f"{'.'.join(str(x) for x in error.absolute_path) or 'body'}: {error.message}"
            for error in errors[:12]
        ]
    except Exception as exc:
        return [f"Could not validate request body against the OpenAPI schema: {type(exc).__name__}: {exc}"]


def describe_operation(operation: dict, include_schema: bool = False) -> dict:
    """Return compact operation metadata for the model and UI."""
    result = {
        "operation_id": operation["id"],
        "method": operation["method"],
        "read_only": bool(operation.get("read_only")),
        "path": operation["path"],
        "summary": operation.get("summary", ""),
        "tags": operation.get("tags", []),
        "parameters": [
            {k: p.get(k) for k in ("name", "in", "required", "description", "schema", "style", "explode")}
            for p in operation.get("params", [])
        ],
        "request_body_required": bool((operation.get("request_body") or {}).get("required")),
        "request_body_media_type": (operation.get("request_body") or {}).get("media_type"),
        "request_body_description": (operation.get("request_body") or {}).get("description", ""),
        "operation_description": operation.get("description", ""),
        "deprecated": bool(operation.get("deprecated", False)),
        "security": operation.get("security", []),
        "response_metadata": {
            str(status): {
                "description": response.get("description", ""),
                "media_types": list((response.get("content") or {}).keys()),
            }
            for status, response in (operation.get("responses") or {}).items()
            if isinstance(response, dict)
        },
    }
    if include_schema and operation.get("request_body"):
        result["request_body_schema"] = operation["request_body"].get("schema")
    if include_schema:
        response_details = {}
        for status, response in (operation.get("responses") or {}).items():
            if not str(status).startswith("2") and status != "default":
                continue
            if not isinstance(response, dict):
                continue
            content = response.get("content") or {}
            json_response = content.get("application/json") or {}
            response_details[str(status)] = {
                "description": response.get("description", ""),
                "schema": json_response.get("schema"),
            }
        result["responses"] = response_details
    return result
