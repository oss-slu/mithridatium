import os

from fastmcp import FastMCP

mcp = FastMCP("mithridatium")

def main():
    # load data in-process; forked DataLoader workers deadlock under the MCP server
    os.environ["MITHRIDATIUM_NUM_WORKERS"] = "0"
    from mithridatium.mcp import audit_mcp, detect_mcp, repair_mcp

    # stdio servers are started by the agents; keep their stderr to warnings only
    mcp.run(transport="stdio", show_banner=False, log_level="WARNING")
