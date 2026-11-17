# NinjaOne Local Expert Agent

Local browser UI for Microsoft Foundry Local with two NinjaOne backends:
- **MCP mode**: launches the local NinjaOne API Gateway as an MCP STDIO subprocess by default. Streamable HTTP remains an optional transport for separately hosted MCP servers.
- **Native mode**: uses the bundled NinjaOne API v2 OpenAPI specification directly; no MCP server is required.

## Foundry endpoint and model discovery

The app discovers the running Foundry Local endpoint and loaded models. It runs `foundry server status`, queries the discovered `/v1/models` endpoint, and populates the model selector. Leave `FOUNDRY_BASE_URL` blank for automatic discovery. Use **Refresh Foundry models** after starting/loading models or restarting Foundry.

## Optional OpenAI-compatible LLM Gateway

Foundry Local and a remote OpenAI-compatible gateway can be used side by side. The provider selector in the browser UI chooses the inference provider for each chat request; it does not change the Native/MCP backend or NinjaOne API permissions.

Configure the gateway in your local, untracked `.env` file:

```dotenv
LLM_PROVIDER=foundry
LLM_GATEWAY_BASE_URL=https://llm-gateway.example.com/v1
LLM_GATEWAY_API_KEY=your_gateway_token
# Optional path to your company's PEM-encoded root/intermediate CA bundle
LLM_GATEWAY_CA_BUNDLE=C:/certificates/company-root-ca.pem
LLM_GATEWAY_MODEL=your_model_id
# Optional, for vision models with non-descriptive IDs:
LLM_GATEWAY_VISION_MODELS=model-id-1,model-id-2
```

Set `LLM_GATEWAY_BASE_URL` to the API base URL (normally ending in `/v1`), not the full `/chat/completions` route. The agent queries `/models` to populate the model selector. `LLM_GATEWAY_MODEL` is an optional fallback when the gateway does not expose model discovery.

The gateway must accept OpenAI-compatible `/chat/completions` requests with Bearer-token authentication. Tool/function calling is required for model-driven NinjaOne operations in both Native and MCP modes. Vision requests additionally require a gateway vision model. Capability is detected from model IDs/metadata where possible; set `LLM_GATEWAY_VISION_MODELS` to a comma-separated list when the gateway uses model IDs that do not identify vision support.

For an internally issued TLS certificate, export the approved company root/intermediate CA certificate in PEM format and set `LLM_GATEWAY_CA_BUNDLE` to its local file path (for example, `C:/certificates/company-root-ca.pem`). This CA bundle is used for both `/models` discovery and inference requests to the configured gateway. Leave it blank for the normal system trust store. Do not use `verify=False` or disable TLS verification.

The gateway token is read only by the backend and is never returned by the configuration endpoint or sent to browser JavaScript. Keep `.env` out of Git. Gateway inference sends prompts and any included NinjaOne context to that gateway, so confirm its data-handling policy before using it with production information. Foundry Local remains available as the local/private option.

Verify discovery in PowerShell:

```powershell
foundry server status
```

## Configuration

Copy `.env.example` to `.env` and configure:
- `FOUNDRY_BASE_URL`: optional explicit override; blank enables automatic discovery.
- `MCP_TRANSPORT`: `stdio` (default, launches the local gateway) or `streamable_http`.
- `NINJA_MCP_SERVER_DIR`: local path to the separately cloned `NinjaRMM-API-Gateway-MCP` repository; defaults to a sibling folder beside `N1_local_agent`.
- `NINJA_MCP_PYTHON`: optional explicit path to the MCP repository's virtual-environment Python executable.
- `MCP_URL`: required only when `MCP_TRANSPORT=streamable_http`.
- `AGENT_BACKEND`: `mcp` or `native`.
- `NINJA_BASE_URL`, `NINJA_CLIENT_ID`, `NINJA_CLIENT_SECRET`: shared NinjaOne credentials. The MCP launcher maps them to the gateway's `NINJAONE_*` variable names.
- `NINJA_TOKEN_URL`: optional token endpoint override; the default is `/oauth/token` on your regional host, with a fallback to the legacy `/ws/oauth/token` endpoint when the default returns 404/405.
- `NINJA_SCOPE`: NinjaOne OAuth scopes. `monitoring` is read-only; approved write actions require `monitoring management`.

For write support, enable **Management** in the NinjaOne OAuth application and set:

```dotenv
NINJA_SCOPE=monitoring management
```

The OAuth app must actually be granted those scopes. Changing `.env` cannot grant permissions that are not enabled in NinjaOne. NinjaOne documents Management as allowing device/organization changes and script execution, so protect the client secret and keep the app bound to the smallest scope set required. The Control scope is for remote access and is not requested by this agent.

## Start

1. Start Foundry Local and load a chat model.
2. For MCP mode, clone `amitdabhade43/NinjaRMM-API-Gateway-MCP` into the sibling folder configured by `NINJA_MCP_SERVER_DIR`, then install its dependencies in its own virtual environment:
 
```powershell
cd ..
git clone https://github.com/amitdabhade43/NinjaRMM-API-Gateway-MCP.git
cd .\NinjaRMM-API-Gateway-MCP
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
cd ..\N1_local_agent
```

   The Local Agent launches the gateway automatically over STDIO; you do not need to start a separate MCP HTTP server.
3. From the Local Agent project directory run:

```powershell
.\run.ps1
```

4. Open `http://127.0.0.1:8080`.
5. Select a loaded model and ask a NinjaOne question.

## Request routing and performance

Chat requests first check for a safe deterministic route before resolving or invoking an LLM:
- Native mode can answer supported fast-path requests directly through the NinjaOne API, before any LLM provider discovery.
- MCP servers that expose resource-specific list tools can answer supported simple inventory requests directly through the cached tool catalog, before provider discovery.
- The schema-driven `NinjaRMM-API-Gateway-MCP` is also supported: the orchestrator can use its generic `discover_endpoints`, `inspect_endpoint` and GET execution tools directly for simple, single-purpose reads, without resolving or invoking the main model. The MCP server skips its secondary semantic-model call for those simple discovery intents. Filtered, cursor-paginated, contextual and multi-step requests continue through the model.
- Requests that do not qualify continue through the selected model and the relevant Native or MCP execution adapter.

Foundry discovery results are cached for 30 seconds after success and 3 seconds after failure. Gateway discovery results are cached for 30 seconds after success and 8 seconds after failure. The Foundry and Gateway refresh actions bypass these caches. Gateway cache durations can be overridden with `LLM_GATEWAY_DISCOVERY_CACHE_SECONDS` and `LLM_GATEWAY_ERROR_CACHE_SECONDS`.

Execution activity includes elapsed-time metrics for direct MCP calls, Native model inference and Native NinjaOne tool calls. The server log records total request duration and MCP model-agent duration. These measurements help distinguish model latency from NinjaOne API latency.

**MCP server boundary:** the gateway is a separate repository and Python environment. In the default STDIO mode, this app launches it as a subprocess and passes only the required NinjaOne, approval and semantic-model settings. The gateway owns API discovery, OpenAPI validation, pagination and HTTP execution.

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

For count-only questions such as "total device access count of a user", the Native backend returns the count from the complete `accessibleDeviceIds` array and does not fetch or print the device inventory. The complete ID array is preserved (not truncated to 50) for every resolved end-user record. List requests still resolve device names and display results in groups of 20.

Person-access reporting follows the fields actually exposed by the API: the agent resolves end users through `GET /v2/user/end-users`, preserves the complete `accessibleDeviceIds` list (not just the first 50), and displays long lists in groups of 20. `assignedDeviceIds` are shown separately and are never counted as accessible devices. The documented `PATCH /v2/user/end-user/{id}/device-access` operation can add or remove up to 100 accessible device IDs per request; the agent must check current access first, avoid re-adding IDs already present, and prepare changes for explicit UI approval. This direct device-access operation is for end users only; the published spec does not expose the equivalent operation for technicians. `GET /v2/users?includeRoles=true` provides user device IDs and role names. Technician roles and members are readable, but this specification has no GET endpoint for reading role `organizationPermissions` (only a PATCH setter). The agent must not invent technician organization-access totals.

**Current transport limitation:** five multipart operations are catalogued but not yet executable through the JSON chat interface: `uploadTempAttachments`, `uploadKnowledgeBaseArticles`, `createRelatedItem`, `createComment`, and `updateScanGroup` (CSV upload). They are not silently converted into JSON; a session-bound file attachment workflow is required.

## MCP backend

Set `AGENT_BACKEND=mcp`. With `MCP_TRANSPORT=stdio`, the app launches `python -m ninja_mcp` from the separate gateway repository using that repository's own `.venv`. The default sibling-folder layout is:

```text
C:\project2\
  N1_local_agent\
  NinjaRMM-API-Gateway-MCP\
```

The launcher reuses the existing Native `.env` as the source of truth:
- `NINJA_CLIENT_ID` and `NINJA_CLIENT_SECRET` are mapped to `NINJAONE_CLIENT_ID` and `NINJAONE_CLIENT_SECRET`.
- `NINJA_BASE_URL` is mapped to the regional `NINJAONE_INSTANCE` hostname.
- `NINJA_SCOPE` is mapped to `NINJAONE_SCOPE`.
- The active chat model ID and resolved provider endpoint are passed to the gateway's optional semantic endpoint ranker. The endpoint follows dynamic Foundry discovery; it is not tied to a hardcoded port.
- Optional `NINJA_MCP_LLM_BASE_URL` and `NINJA_MCP_LLM_MODEL` values override the shared active model configuration.

For approved writes, set `NINJA_SCOPE=monitoring management` and configure `NINJA_MCP_APPROVAL_SECRET` in the Local Agent's untracked `.env`. The same secret is passed only to the trusted MCP subprocess: the Local Agent signs a short-lived token after the user approves, and the gateway verifies its signature, expiry, method, path and canonical request-body digest. The secret is never sent to the model. POST, PUT and PATCH are held for approval; DELETE is blocked. The Local Agent also checks the Management scope before preparing and executing an MCP write. NinjaOne's OAuth application and user role permissions remain authoritative.

The MCP gateway has its own Python environment. Install it once as shown in **Start** above. Do not copy secrets into the gateway repository or commit either repository's `.env` file.

If connecting to an already-hosted MCP server instead, set `MCP_TRANSPORT=streamable_http` and configure `MCP_URL`. The remote server must expose Streamable HTTP transport. The local gateway in `NinjaRMM-API-Gateway-MCP` uses STDIO by default.

The app discovers tools exposed by the gateway. Its available capabilities are limited to the operations that gateway actually exposes; the OpenAPI gateway supports schema-driven GET/POST/PUT/PATCH operations, with DELETE blocked.

## Notes

- Qwen/VL screenshot uploads are enabled only when the selected Foundry model is identified as vision-capable.
- Chat and pending-approval state are held in backend memory and cleared when the backend restarts.
- Keep the web server bound to loopback unless authentication and network protections are added.
- Local model tool-calling reliability varies; the Native backend uses deterministic API discovery and schema validation to reduce dependence on model-generated tool syntax.
