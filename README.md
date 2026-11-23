# NinjaOne Local Expert Agent

Local browser UI for Microsoft Foundry Local with two NinjaOne backends:
- **MCP mode**: connects to the configured streamable HTTP MCP server.
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

## Request routing and performance

Chat requests first check for a safe deterministic route before resolving or invoking an LLM:
- Native mode can answer supported fast-path requests directly through the NinjaOne API, before any LLM provider discovery.
- MCP servers that expose resource-specific list tools can answer supported simple inventory requests directly through the cached tool catalog, before provider discovery.
- The schema-driven `NinjaRMM-API-Gateway-MCP` is also supported: the orchestrator can use its generic `discover_endpoints`, `inspect_endpoint` and GET execution tools directly for simple, single-purpose reads, without resolving or invoking the main model. The MCP server skips its secondary semantic-model call for those simple discovery intents. Filtered, cursor-paginated, contextual and multi-step requests continue through the model.
- Requests that do not qualify continue through the selected model and the relevant Native or MCP execution adapter.

Foundry discovery results are cached for 30 seconds after success and 3 seconds after failure. Gateway discovery results are cached for 30 seconds after success and 8 seconds after failure. The Foundry and Gateway refresh actions bypass these caches. Gateway cache durations can be overridden with `LLM_GATEWAY_DISCOVERY_CACHE_SECONDS` and `LLM_GATEWAY_ERROR_CACHE_SECONDS`.

Execution activity includes elapsed-time metrics for direct MCP calls, Native model inference and Native NinjaOne tool calls. The server log records total request duration and MCP model-agent duration. These measurements help distinguish model latency from NinjaOne API latency.

**MCP server boundary:** this repository contains the MCP client, routing and UI integration. The configured `MCP_URL` points to a separately hosted MCP server; changes here improve client-side discovery, routing and measurement, but do not change the server's own implementation or internal API pagination.

## Native OpenAPI backend

Set `AGENT_BACKEND=native`. The bundled `ninja_spec.json` is the source of truth for executable endpoint discovery and request validation. The bundled `ninjaOne-API-core-resources.yaml` is loaded as a supplemental schema catalog; it enriches field discovery without adding or overriding executable operations.

Reference sources: [NinjaOne Public API v2 JSON](https://app.ninjarmm.com/apidocs/NinjaRMM-API-v2.json) and [supplemental NinjaOne core-resources YAML](https://github.com/amitdabhade43/N1_local_agent/blob/feature/ninjaone-spec-field-awareness/ninjaOne-API-core-resources.yaml). Review both when changing endpoint routing, request/response validation, schema discovery, role membership or device-access behavior.
To compare or refresh it from NinjaOne's official OpenAPI URL, run `python scripts/sync_ninjaone_api_catalog.py --check` (check only) or `python scripts/sync_ninjaone_api_catalog.py` (safe atomic update). The sync script validates OpenAPI format and catalog size and refuses an unexpected catalog downgrade unless explicitly overridden. The agent can also refresh on startup with `NINJA_SPEC_REFRESH=1`; it validates the downloaded document and replaces the file atomically.

The current specification contains 315 operations across 254 paths:
- **416 component schemas** and **2,485 top-level schema properties** are bundled. The published NinjaOne API documentation currently lists the same 315 operations and 416 models (Public API 2.0, `2.0.9-draft`).
- **151 GET** read operations.
- **91 POST**, **30 PATCH**, and **20 PUT** operations in the source specification.
- **23 DELETE** operations are blocked.
- A further **7 POST operations explicitly perform delete actions** and are also blocked.
- The resulting catalog exposes **285 operations**: 151 GET, 84 POST, 30 PATCH and 20 PUT.

Capabilities:
- Search API operations by endpoint name, description, tags, parameters, and request/response fields.
- Search every component-schema field and every operation parameter, request-body field and response field with `search_fields`; results include exact field paths, types, descriptions, required flags, enums, formats and read/write constraints where published.
- `inspect_api` returns a lossless field catalog for an operation, paginated when necessary. It does not silently stop at the first 35 properties or depth four; use its `schema_fields_next_offset` to retrieve remaining fields.
- Search/indexing traverses nested schemas and enum values deeply, including array item objects and composed schemas.
- Response shaping preserves nested non-secret fields and supports dotted field selection/filtering. Credential-like values (passwords, tokens, secrets, keys and authorization values) are redacted.
- Inspect the exact operation schema, required parameters, enums, and request body before calling it.
- Execute validated GET requests and POST operations explicitly documented as read-only queries (for example, listing tickets by board); filter, count, group, and paginate where the OpenAPI request/response schemas define an integer ID cursor, opaque nextCursor/pageInfo cursor, or ticket-board lastCursorId cursor; unknown pagination is marked partial.
- Prepare state-changing POST/PUT/PATCH requests using OpenAPI request schemas. The agent displays the exact method, path, query and body; it does not send the mutation until you click **Approve and execute**.
- Approval is tied to the originating chat session, expires after 10 minutes, and can be used once. Rejecting or clearing the conversation discards the pending action.
- Delete operations are excluded from discovery and blocked again at the HTTP execution layer. This includes POST endpoints whose path or operation description explicitly indicates deletion.

Every write requires both the Management OAuth scope and an explicit user approval. NinjaOne's own role permissions remain authoritative. Script execution, reboot, patch actions, decommissioning, policy changes and other state-changing operations are treated as writes and require approval.

For count-only questions such as "total device access count of a user", the Native backend returns the count from the complete `accessibleDeviceIds` array and does not fetch or print the device inventory. The complete ID array is preserved (not truncated to 50) for every resolved end-user record. List requests still resolve device names and display results in groups of 20.

Technician account type and technician-role membership are distinct API concepts. `GET /v2/users?userType=TECHNICIAN` filters user records by account type; `GET /v2/user/roles?roleType=TECHNICIAN` returns technician-type roles with their `members` array. The Native count recipe for the wording "users in technician role" counts distinct member IDs across those roles (so a user assigned to multiple technician roles is counted once) and discloses partial role-inventory results instead of claiming a complete total. Do not substitute the technician account count for the technician-role membership count without saying which metric is being reported.

Person-access reporting follows the fields actually exposed by the API: the agent resolves end users through `GET /v2/user/end-users`, preserves the complete `accessibleDeviceIds` list (not just the first 50), and displays long lists in groups of 20. `assignedDeviceIds` are shown separately and are never counted as accessible devices. The documented `PATCH /v2/user/end-user/{id}/device-access` operation can add or remove up to 100 accessible device IDs per request; the agent must check current access first, avoid re-adding IDs already present, and prepare changes for explicit UI approval. This direct device-access operation is for end users only; the published spec does not expose the equivalent operation for technicians. `GET /v2/users?includeRoles=true` provides user device IDs and role names. Technician roles and members are readable, but this specification has no GET endpoint for reading role `organizationPermissions` (only a PATCH setter). The agent must not invent technician organization-access totals.

**Current transport limitation:** five multipart operations are catalogued but not yet executable through the JSON chat interface: `uploadTempAttachments`, `uploadKnowledgeBaseArticles`, `createRelatedItem`, `createComment`, and `updateScanGroup` (CSV upload). They are not silently converted into JSON; a session-bound file attachment workflow is required.

## MCP backend

Set `AGENT_BACKEND=mcp`. The app discovers tools exposed by the configured MCP server. Its available capabilities are limited to the tools that server actually exposes; the current local MCP server may expose fewer operations than the full NinjaOne API specification.

## Notes

- Qwen/VL screenshot uploads are enabled only when the selected Foundry model is identified as vision-capable.
- Chat and pending-approval state are held in backend memory and cleared when the backend restarts.
- Keep the web server bound to loopback unless authentication and network protections are added.
- Local model tool-calling reliability varies; the Native backend uses deterministic API discovery and schema validation to reduce dependence on model-generated tool syntax.


## Unified spec-driven agent workflow

Native and MCP are execution interfaces to one user-facing agent workflow. The bundled ninja_spec.json is the executable source of truth: the agent searches the operation catalog, inspects the selected contract and fields, resolves entity IDs, validates parameters and request bodies, and uses the resulting live data to answer.

- Reads: GET and specification-confirmed read-only POST queries may run directly. Pagination is followed only when the OpenAPI request and response schemas establish a safe cursor. Counts and lists disclose incomplete pagination or display truncation.
- Writes: documented non-delete JSON and URL-encoded POST, PUT and PATCH operations are approval-gated. The exact method, path, query and body are presented before execution. A chat message such as "yes" does not approve a write; the user must press Approve and execute.
- Safety: DELETE and explicit delete operations remain blocked. Ambiguous targets, missing required fields, unsupported payloads and incomplete identity searches fail closed rather than guessing.
- MCP approval: every call/raw_call is matched to the same local JSON OpenAPI specification and its parameters/body are validated before execution. The Local Agent intercepts the NinjaOne gateway's call/raw_call tools and compatible execute_endpoint tools. Write requests are held server-side and invoked only after the approval button is clicked; the upstream call/raw_call tools do not accept an approval-token argument.
- Native tool use: requests that need live NinjaOne evidence trigger a tool-call retry if the model initially responds with prose. The Native backend requests tool_choice=required where the OpenAI-compatible local provider supports it, and otherwise falls back to an explicit tool-call instruction with fail-closed response checks.
- Coverage limits: multipart and file-upload operations can be discovered in the specification but are not executed through the current chat transport. The agent must explain this boundary instead of claiming that an upload occurred.

Simple unfiltered collection counts and lists can be routed from the loaded OpenAPI catalog without hard-coded endpoint tables. Filtered, relational, entity-specific and investigative requests use the full tool-using workflow.


## Generic execution planning, verification and progress evidence

The Native backend now creates a structured execution outline for each model-routed request using the loaded OpenAPI operation index. The outline records the request intent (read, mutation, or blocked destructive action), the route selected, the top specification-ranked candidate operations, their HTTP methods and paths, whether each is read-only or approval-gated, and the safety constraints that apply. This is an inspectable execution summary, not private model chain-of-thought.

The Native activity stream includes an `execution_plan` event before model tool execution. Each live `call_api` and `resolve_entity` result also receives a separate `verification` event. The shared result verifier checks the JSON response envelope, required result shape, explicit API/validation errors, approval-prepared state, and partial/truncated pagination markers. A partial result is never labelled complete. A prepared write is explicitly distinguished from an executed write.

This is a first reusable verification layer, not proof that every business-level interpretation is correct. The model must still match returned fields and filters to the user's request; write completion must still be established from the approval action's NinjaOne response. The verifier does not infer missing fields or silently promote a partial result to a complete tenant-wide count.

Deterministic recipes remain as fast paths for narrow, high-confidence workflows. They should be treated as optimizations over the common specification-driven engine, not as the only way the agent can support an API operation. New resource support should preferentially use the OpenAPI operation/field index, shared entity resolution, request validation, pagination handling and shared verification instead of adding another question-specific regex recipe.
