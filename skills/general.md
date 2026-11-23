---
name: general
description: General spec-driven NinjaOne workflow
agents: general
---
# General spec-driven workflow

- Treat ninja_spec.json as the executable API contract; supplemental YAML is schema-discovery help only.
- Search operations and fields, inspect exact contracts, resolve entity IDs, validate arguments and verify results.
- Prefer the shared generic tool workflow over adding a new question-specific regex recipe.
- Reads may execute only when the operation is documented as read-only.
- Every state-changing POST, PUT or PATCH requires explicit UI approval.
- DELETE and explicitly destructive operations are always blocked.
