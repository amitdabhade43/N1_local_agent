#!/usr/bin/env python3
"""Safely synchronize the bundled NinjaOne OpenAPI specification from NinjaOne."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import requests

SPEC_URL = "https://app.ninjarmm.com/apidocs/NinjaRMM-API-v2.json"
METHODS = {"get", "post", "put", "patch", "delete"}


def summary(spec: dict) -> dict:
    paths = spec.get("paths") or {}
    operations = [
        (method.lower(), path, operation)
        for path, item in paths.items()
        if isinstance(item, dict)
        for method, operation in item.items()
        if method.lower() in METHODS and isinstance(operation, dict)
    ]
    schemas = spec.get("components", {}).get("schemas") or {}
    return {
        "version": (spec.get("info") or {}).get("version", "unknown"),
        "openapi": spec.get("openapi", "unknown"),
        "operations": len(operations),
        "paths": len(paths),
        "schemas": len(schemas),
        "properties": sum(
            len(schema.get("properties") or {})
            for schema in schemas.values()
            if isinstance(schema, dict)
        ),
        "methods": {
            method.upper(): sum(1 for item in operations if item[0] == method)
            for method in sorted(METHODS)
        },
        "tags": {
            tag: sum(1 for _, _, operation in operations if tag in (operation.get("tags") or []))
            for tag in sorted({
                tag for _, _, operation in operations
                for tag in (operation.get("tags") or [])
            })
        },
    }


def validate(spec: dict, current: dict | None, allow_downgrade: bool) -> None:
    if not isinstance(spec, dict) or not str(spec.get("openapi", "")).startswith("3."):
        raise ValueError("The download is not a valid OpenAPI 3.x document.")
    new = summary(spec)
    if new["operations"] < 100 or new["schemas"] < 100:
        raise ValueError(
            f"Downloaded specification is incomplete: {new['operations']} operations, "
            f"{new['schemas']} schemas."
        )
    if current and not allow_downgrade:
        old = summary(current)
        if new["operations"] < old["operations"] or new["schemas"] < old["schemas"]:
            raise ValueError(
                "Refusing a smaller upstream catalog without --allow-downgrade. "
                f"Current={old['operations']} operations/{old['schemas']} schemas; "
                f"downloaded={new['operations']} operations/{new['schemas']} schemas."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check for updates without changing the file.")
    parser.add_argument("--allow-downgrade", action="store_true", help="Allow a smaller upstream catalog.")
    parser.add_argument("--url", default=SPEC_URL, help="OpenAPI JSON URL.")
    parser.add_argument("--file", default=str(Path(__file__).resolve().parents[1] / "ninja_spec.json"))
    args = parser.parse_args()

    target = Path(args.file).resolve()
    current = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
    response = requests.get(args.url, timeout=(15, 120))
    response.raise_for_status()
    downloaded = response.json()
    validate(downloaded, current, args.allow_downgrade)

    old_summary = summary(current) if current else None
    new_summary = summary(downloaded)
    old_bytes = target.read_bytes() if target.exists() else b""
    new_bytes = json.dumps(downloaded, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    old_hash = hashlib.sha256(old_bytes).hexdigest() if old_bytes else None
    new_hash = hashlib.sha256(new_bytes).hexdigest()

    print("Current bundled specification:", old_summary or "not present")
    print("Official NinjaOne specification:", new_summary)
    print("SHA-256:", new_hash)
    if old_hash == new_hash:
        print("Already synchronized; no file change needed.")
        return 0
    if args.check:
        print("Update available. Run without --check to replace ninja_spec.json.")
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="ninja_spec.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(new_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    print(f"Updated {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
