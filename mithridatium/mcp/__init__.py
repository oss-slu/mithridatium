from fastmcp import FastMCP

mcp = FastMCP("mithridatium")

def main() -> None:
    from mithridatium.mcp import audit_mcp
    
    mcp.run(transport="stdio")
