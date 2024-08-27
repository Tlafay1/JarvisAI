# BrowserAgent.py
import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import (
    SendTool,
)
from langroid.pydantic_v1 import Field
from langroid.agent.tool_message import ToolMessage
import webbrowser


class OpenWebsiteTool(ToolMessage):
    request = "open_website"
    purpose = "To open <url> in a new browser tab."
    url: str = Field(..., description="URL of the website to open")

    def handle(self) -> str:
        webbrowser.open_new_tab(self.url)
        return SendTool(to="User", content=f"Opened website '{self.url}'")


class SearchOnGoogleTool(ToolMessage):
    request = "search_on_google"
    purpose = "To perform a Google search based on <query>."
    query: str = Field(..., description="Query to search on Google")

    def handle(self) -> str:
        webbrowser.open_new_tab(f"https://www.google.com/search?q={self.query}")
        return SendTool(to="User", content=f"Opened Google search for '{self.query}'")


class BrowserAgent:

    def __init__(self, llm_config: lm.OpenAIGPTConfig):
        config = lr.ChatAgentConfig(
            name="browser_agent",
            llm=llm_config,
            system_message="""
                You are an expert on searching on Google and opening websites.
                You will receive an instruction and then execute it using the appropriate tool.
                """,
        )

        self.agent = lr.ChatAgent(config)
        self.agent.enable_message(
            [SearchOnGoogleTool, OpenWebsiteTool], use=True, handle=False
        )
        self.task = lr.Task(self.agent, single_round=True)
