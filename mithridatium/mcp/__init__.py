from fastmcp import FastMCP

mcp = FastMCP("mithridatium")

def main():
    from mithridatium.mcp import audit_mcp

    mcp.run(transport="stdio")
