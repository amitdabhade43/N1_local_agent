# NinjaOne Local Expert Agent

Local browser UI for Foundry Local models and your existing NinjaOne MCP server.

## Configuration
- Foundry API: `http://127.0.0.1:52072/v1`
- MCP endpoint: `http://127.0.0.1:8000/mcp`
- Qwen model: `qwen3-vl-4b-instruct-generic-cpu:3`
- Phi model: set the exact ID returned by Foundry `/v1/models`.

## Start
1. Start your existing NinjaOne MCP server first.
2. Ensure the selected Foundry model is loaded and the Foundry API is running.
3. Run `.un.ps1`
4. Open `http://127.0.0.1:8080`

Edit `.env` to set the exact Phi model ID and other endpoints.

## Notes
- The app currently exposes only your read-only MCP tools.
- Qwen screenshot uploads are sent as a data URL to the local Foundry OpenAI-compatible chat endpoint.
- Chat session state is held in backend memory and is cleared when the backend restarts.
- This starter uses request/response chat; tool activity is displayed after each agent turn.
- Bind the web server to loopback only unless you add authentication and network protections.
- Validate model tool-call compatibility with both model variants.
