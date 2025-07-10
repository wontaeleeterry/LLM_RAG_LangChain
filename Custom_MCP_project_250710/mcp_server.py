from typing import List
from mcp.server.fastmcp import FastMCP

server = FastMCP("MCP Menu Recommender Server")

@server.tool()
def recommend_menu(preference: str) -> str:
    """
    Recommend a menu item based on user preference.

    Args:
        preference (str): User preference, must be one of {"vegetarian", "sweet", "none"}.
    """

    if "vegetarian" in preference:
        return (
            "Today, I recommend a Caesar Salad for a fresh, vegetarian-friendly option!"
        )
    elif "sweet" in preference:
        return "Today, I recommend a Tiramisu for a delicious sweet treat!"
    else:
        return "Today, I recommend a classic Margherita Pizza!"

if __name__ == "__main__":
    server.run()