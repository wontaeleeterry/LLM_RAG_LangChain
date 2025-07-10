import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# model = ChatOpenAI(model="gpt-4o")

# 이 부분을 로컬(LM Tools)로 변경하면,##################################
model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    # model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    model = "lmstudio-community/gemma-2-2b-it-GGUF",
    temperature=0.1,
    streaming=True,
    #callbacks=[StreamingStdOutCallbackHandler()], # 스트림 출력 콜백
)
###################################################################

server_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
server_params = StdioServerParameters(
    command="python",
    args=[server_py],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session: # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(
                model,
                tools,
                prompt="You are a helpful assistant. Answer the user's questions in one word.",
            )
            agent_response = await agent.ainvoke(
                {"messages": "I am a vegetarian. Can you recommend a menu for me?"}
            )
            print(agent_response)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())