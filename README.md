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
- `NINJA_BASE_URL`, `NINJA_CLIENT_ID`, `NINJA_CLIENT_SECRET`: required for Native mode.
- `NINJA_SCOPE`: NinjaOne OAuth scopes. `monitoring` is read-only; approved write actions require `monitoring management`.

For write support, enable **Management** in the NinjaOne OAuth application and set:

```dotenv
NINJA_SCOPE=monitoring management
```

The OAuth app must actually be granted those scopes. Changing `.env` cannot grant permissions that are not enabled in NinjaOne.

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
- Execute validated GET requests; filter, count, group, and paginate only where the OpenAPI parameters confirm an integer ID cursor.
- Prepare POST/PUT/PATCH requests using OpenAPI request schemas. The agent displays the exact method, path, query and body; it does not send the mutation until you click **Approve and execute**.
- Approval is tied to the originating chat session, expires after 10 minutes, and can be used once. Rejecting or clearing the conversation discards the pending action.
- Delete operations are excluded from discovery and blocked again at the HTTP execution layer. This includes POST endpoints whose path or operation description explicitly indicates deletion.

Every write requires both the Management OAuth scope and an explicit user approval. NinjaOne's own role permissions remain authoritative. Script execution, reboot, patch actions, decommissioning, policy changes and other state-changing operations are treated as writes and require approval.

**Current transport limitation:** operations requiring multipart or binary file uploads are identified in the spec but cannot yet be submitted through the JSON chat interface. These are not silently converted into JSON.

## MCP backend

Set `AGENT_BACKEND=mcp`. The app discovers tools exposed by the configured MCP server. Its available capabilities are limited to the tools that server actually exposes; the current local MCP server may expose fewer operations than the full NinjaOne API specification.

## Notes

- Qwen/VL screenshot uploads are enabled only when the selected Foundry model is identified as vision-capable.
- Chat and pending-approval state are held in backend memory and cleared when the backend restarts.
- Keep the web server bound to loopback unless authentication and network protections are added.
- Local model tool-calling reliability varies; the Native backend uses deterministic API discovery and schema validation to reduce dependence on model-generated tool syntax.
