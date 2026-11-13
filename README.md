# NinjaOne Local Expert Agent

Local browser UI for Microsoft Foundry Local with two NinjaOne backends:
- **MCP mode**: connects to the configured streamable HTTP MCP server.
- **Native mode**: uses the bundled NinjaOne API v2 OpenAPI specification directly; no MCP server is required.

## Foundry endpoint and model discovery

The app discovers the running Foundry Local endpoint and loaded models. It runs `foundry server status`, queries the discovered `/v1/models` endpoint, and populates the model selector. Leave `FOUNDRY_BASE_URL` blank for automatic discovery. Use **Refresh Foundry models** after starting/loading models or restarting Foundry.

Verify discovery in PowerShell:

```powershell
foundry server status
```

## Configuration

Copy `.env.example` to `.env` and configure:
- `FOUNDRY_BASE_URL`: optional explicit override; blank enables automatic discovery.
- `MCP_URL`: streamable HTTP endpoint for MCP mode.
- `AGENT_BACKEND`: `mcp` or `native`.
- `NINJA_BASE_URL`: your NinjaOne regional API host.
- `NINJA_CLIENT_ID`, `NINJA_CLIENT_SECRET`: required for Native mode.
- `NINJA_TOKEN_URL`: optional token endpoint override; the default is `/oauth/token` on your regional host, with a fallback to the legacy `/ws/oauth/token` endpoint when the default returns 404/405.
- `NINJA_SCOPE`: NinjaOne OAuth scopes. `monitoring` is read-only; approved write actions require `monitoring management`.

For write support, enable **Management** in the NinjaOne OAuth application and set:

```dotenv
NINJA_SCOPE=monitoring management
```

The OAuth app must actually be granted those scopes. Changing `.env` cannot grant permissions that are not enabled in NinjaOne. NinjaOne documents Management as allowing device/organization changes and script execution, so protect the client secret and keep the app bound to the smallest scope set required. The Control scope is for remote access and is not requested by this agent.

## Start

1. Start Foundry Local and load a chat model.
2. In MCP mode, start your MCP server.
3. From the project directory run:

```powershell
.\run.ps1
```

4. Open `http://127.0.0.1:8080`.
5. Select a loaded model and ask a NinjaOne question.

## Native OpenAPI backend

Set `AGENT_BACKEND=native`. The bundled `ninja_spec.json` is the source of truth for endpoint discovery and request validation.

The current specification contains 315 operations across 254 paths:
- **151 GET** read operations.
- **91 POST**, **30 PATCH**, and **20 PUT** operations in the source specification.
- **23 DELETE** operations are blocked.
- A further **7 POST operations explicitly perform delete actions** and are also blocked.
- The resulting catalog exposes **285 operations**: 151 GET, 84 POST, 30 PATCH and 20 PUT.

Capabilities:
- Search API operations by endpoint name, description, tags, parameters, and request/response fields.
- Inspect the exact operation schema, required parameters, enums, and request body before calling it.
- Execute validated GET requests and POST operations explicitly documented as read-only queries (for example, listing tickets by board); filter, count, group, and paginate where the OpenAPI request/response schemas define an integer ID cursor, opaque nextCursor/pageInfo cursor, or ticket-board lastCursorId cursor; unknown pagination is marked partial.
- Prepare state-changing POST/PUT/PATCH requests using OpenAPI request schemas. The agent displays the exact method, path, query and body; it does not send the mutation until you click **Approve and execute**.
- Approval is tied to the originating chat session, expires after 10 minutes, and can be used once. Rejecting or clearing the conversation discards the pending action.
- Delete operations are excluded from discovery and blocked again at the HTTP execution layer. This includes POST endpoints whose path or operation description explicitly indicates deletion.

Every write requires both the Management OAuth scope and an explicit user approval. NinjaOne's own role permissions remain authoritative. Script execution, reboot, patch actions, decommissioning, policy changes and other state-changing operations are treated as writes and require approval.

Person-access reporting follows the fields actually exposed by the API: the agent resolves end users through `GET /v2/user/end-users`, preserves the complete `accessibleDeviceIds` list (not just the first 50), and displays long lists in groups of 20. `assignedDeviceIds` are shown separately and are never counted as accessible devices. The documented `PATCH /v2/user/end-user/{id}/device-access` operation can add or remove up to 100 accessible device IDs per request; the agent must check current access first, avoid re-adding IDs already present, and prepare changes for explicit UI approval. This direct device-access operation is for end users only; the published spec does not expose the equivalent operation for technicians. `GET /v2/users?includeRoles=true` provides user device IDs and role names. Technician roles and members are readable, but this specification has no GET endpoint for reading role `organizationPermissions` (only a PATCH setter). The agent must not invent technician organization-access totals.

**Current transport limitation:** five multipart operations are catalogued but not yet executable through the JSON chat interface: `uploadTempAttachments`, `uploadKnowledgeBaseArticles`, `createRelatedItem`, `createComment`, and `updateScanGroup` (CSV upload). They are not silently converted into JSON; a session-bound file attachment workflow is required.

## MCP backend

Set `AGENT_BACKEND=mcp`. The app discovers tools exposed by the configured MCP server. Its available capabilities are limited to the tools that server actually exposes; the current local MCP server may expose fewer operations than the full NinjaOne API specification.

## Notes

- Qwen/VL screenshot uploads are enabled only when the selected Foundry model is identified as vision-capable.
- Chat and pending-approval state are held in backend memory and cleared when the backend restarts.
- Keep the web server bound to loopback unless authentication and network protections are added.
- Local model tool-calling reliability varies; the Native backend uses deterministic API discovery and schema validation to reduce dependence on model-generated tool syntax.
