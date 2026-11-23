---
name: ticketing
description: Ticket and service-desk investigation
agents: tickets
---
# Ticket and service-desk investigation

- Resolve the relevant ticket board before querying board-scoped records.
- Inspect the operation contract and request body before using board query endpoints.
- Follow only documented cursor fields and preserve cursor types exactly.
- Distinguish a page size from a total count; disclose partial results.
- Keep ticket status, priority, assignee and requester fields grounded in the response.
