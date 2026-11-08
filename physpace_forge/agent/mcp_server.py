"""
Model Context Protocol (MCP) JSON-RPC 2.0 Server for PhySpace-Forge (`physpace mcp`).

Allows AI Agents (Claude Desktop, Claude Code, Cursor, Antigravity, Qwen-Agent, OpenHands)
to discover and invoke PhySpace-Forge's continuum physics compiler, 200Hz RT-Patch-IPC
solver, and universal format exporter over standard MCP `stdio` transport.

MCP Configuration (`mcp_config.json` / `claude_desktop_config.json`):
{
  "mcpServers": {
    "physpace-forge": {
      "command": "python",
      "args": ["-m", "physpace_forge.cli", "mcp"]
    }
  }
}
"""

import json
import sys
from typing import Any, Dict

from physpace_forge.agent.tool_schemas import AgentToolDispatcher, PHYSPACE_AGENT_TOOLS


class PhySpaceMCPServer:
    """Standard JSON-RPC 2.0 MCP Server exposing PhySpace-Forge physical tools to AI Agents."""

    def __init__(self) -> None:
        self.dispatcher = AgentToolDispatcher()

    def handle_rpc_request(self, req: Dict[str, Any]) -> Dict[str, Any] | None:
        method = req.get("method", "")
        req_id = req.get("id")
        params = req.get("params") or {}

        # Notifications (no id) do not expect a response
        if req_id is None:
            return None

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "physpace-forge-mcp",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"listChanged": False},
                    "prompts": {"listChanged": False},
                },
            }
        elif method == "tools/list":
            mcp_tools = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "inputSchema": t["function"]["parameters"],
                }
                for t in PHYSPACE_AGENT_TOOLS
            ]
            result = {"tools": mcp_tools}
        elif method == "resources/list":
            result = {"resources": []}
        elif method == "resources/templates/list":
            result = {"resourceTemplates": []}
        elif method == "prompts/list":
            result = {"prompts": []}
        elif method == "tools/call":
            tool_name = str(params.get("name", ""))
            tool_args = params.get("arguments") or {}
            out = self.dispatcher.invoke(tool_name, tool_args)
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(out, indent=2, ensure_ascii=False),
                    }
                ],
                "isError": out.get("status") == "error",
            }
        elif method == "ping":
            result = {}
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }

    def run_stdio_loop(self) -> None:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_rpc_request(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                    sys.stdout.flush()
            except Exception as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse/Execution error: {exc}"},
                }
                sys.stdout.write(json.dumps(err_resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
