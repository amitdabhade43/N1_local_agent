# NinjaOne Local Expert Agent

Local browser UI for Microsoft Foundry Local with two selectable NinjaOne backends:
- **MCP mode** (default): connects to your existing NinjaOne MCP server.
- **Native mode**: uses the NinjaOne API v2 OpenAPI specification directly; no MCP server is required.

## Foundry endpoint and model discovery

The app automatically discovers the running Foundry Local endpoint and loaded models.

1. Start Foundry Local and load the model(s) you want to use.
2. The app runs `foundry server status` to discover the current local endpoint, including its dynamically assigned port.
3. It queries the discovered `/v1/models` endpoint and populates the browser model selector with the model IDs returned by Foundry.
4. Both MCP and Native backends receive the same discovered endpoint and selected model ID.
5. Use **Refresh Foundry models** after starting/loading models or restarting Foundry.

The discovery code also checks the optional `FOUNDRY_BASE_URL` override first, then the CLI-reported endpoint, then legacy ports used by earlier versions of this project. Leave `FOUNDRY_BASE_URL` blank for automatic discovery. The UI only lists models currently returned by Foundry's `/v1/models`; a model must be loaded/available through the local server to be selectable.

The Foundry CLI must be installed and available on the app process `PATH`. Verify discovery manually with:

```powershell
foundry server status
```

## Configuration

Copy `.env.example` to `.env` and configure:

- `FOUNDRY_BASE_URL`: optional explicit override; leave blank for automatic endpoint discovery.
- `MCP_URL`: your existing MCP streamable HTTP endpoint (default `http://127.0.0.1:8000/mcp`).
- `AGENT_BACKEND`: `mcp` or `native`.
- `NINJA_BASE_URL`, `NINJA_CLIENT_ID`, `NINJA_CLIENT_SECRET`: required for Native mode only.

The Foundry endpoint/port and NinjaOne MCP endpoint/port are separate services. Foundry discovery does not change `MCP_URL`.

## Start

1. Start Foundry Local and load a chat model.
2. In MCP mode, start your existing NinjaOne MCP server.
3. In the project directory, run:

   ```powershell
   .\\run.ps1
   ```

4. Open `http://127.0.0.1:8080`.
5. Confirm the Foundry endpoint and loaded models in the left panel. Select a model or click **Refresh Foundry models**.

## Backends

### MCP mode (default)

Set `AGENT_BACKEND=mcp`. The app discovers tools from the configured MCP endpoint and uses the selected Foundry model for the LangChain agent. The currently exposed NinjaOne MCP tools are read-only.

### Native mode (no MCP server)

Set `AGENT_BACKEND=native` plus `NINJA_CLIENT_ID`, `NINJA_CLIENT_SECRET`, and the correct regional `NINJA_BASE_URL`. The native agent uses the same discovered Foundry endpoint/model as MCP mode. It loads NinjaOne API v2 operations from the OpenAPI spec, searches endpoints, paginates, counts, and resolves entities. Native API execution is currently GET-only/read-only.

## Notes

- Qwen/VL screenshot uploads are enabled only when the discovered model ID is identified as vision-capable (for example, a model ID containing `-vl-`).
- Chat session state is held in backend memory and is cleared when the backend restarts.
- Tool activity is displayed after each agent turn.
- Bind the web server to loopback only unless you add authentication and network protections.
- Tool-calling support varies by local model; validate agent tool calls with each model you intend to use.
