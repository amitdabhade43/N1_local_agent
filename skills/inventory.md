---
name: inventory
description: Tenant inventory investigation using the NinjaOne API specification
agents: inventory
---
# Inventory investigation

- Establish the requested tenant/organization scope before counting or listing devices.
- Use the primary JSON OpenAPI catalog to choose executable operations and validate query parameters.
- Follow only documented pagination cursors until completion; report partial results honestly.
- For count-only requests, compute the total from the complete returned collection and avoid dumping every record.
- Preserve stable IDs and distinguish device names, hostnames, serial numbers and organization names.
- Verify the final count and filters against live API results, not the schema catalog.
