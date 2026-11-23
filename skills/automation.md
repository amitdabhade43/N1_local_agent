---
name: automation
description: Scripts, jobs and automation operations
agents: automation
---
# Scripts and automation

- Resolve the exact script, job, target device and organization before proposing an operation.
- Inspect request schemas and enumerate required parameters before constructing a payload.
- Script execution and any other side-effecting operation require explicit UI approval.
- Never assume a script completed because a request was prepared or queued.
- Verify the returned execution/job status after an approved action.
