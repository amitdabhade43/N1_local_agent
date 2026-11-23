---
name: access
description: Device-access and permission reporting with approval-gated changes
agents: access
---
# Device access and permissions

- Distinguish accessibleDeviceIds from assignedDeviceIds; assignment is not proof of access.
- Resolve both the person and device to unambiguous NinjaOne records before preparing a change.
- Device-access PATCH is end-user-only where that is what the published API contract documents; never substitute a same-name technician or console user.
- Check current access first and avoid adding IDs already present.
- Prepare the smallest valid request and require the explicit UI approval action before execution.
- A chat reply such as "yes" is not approval. Never retry a possibly successful write after a timeout without checking its outcome.
