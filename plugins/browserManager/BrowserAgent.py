# BrowserAgent.py
from typing import List
import langroid as lr

from tools import SearchOnGoogleTool, OpenWebsiteTool
from plugin import PluginCore, PluginAgent
from config import LLM_CONFIGS


class BrowserAgent(PluginCore):
    class Meta:
        name = "BrowserAgent"
        description = "A plugin that allows the user to control the web browser."
        version = "0.1"

    def register_agents(self) -> List[PluginAgent] | None:
        config = lr.ChatAgentConfig(
            name=self.Meta.name,
            llm=LLM_CONFIGS.get("small"),
            system_message="""
                You are an expert on controlling the web browser.
                For ANY TASK you receive, you must use the appropriate tool to execute it.
                Once you the TASK is executed, you must send the result back to the user.
                EXTREMELY IMPORTANT: You must NOT execute the TASK yourself, use a tool ONLY.
                """,
        )
        agent = self.Agent(config)
        return agent

    def register_tools(self) -> List[lr.ToolMessage] | None:
        return [SearchOnGoogleTool, OpenWebsiteTool]

    class Agent(PluginAgent):
        def register_tools(self) -> List[lr.ToolMessage] | None:
            return [SearchOnGoogleTool, OpenWebsiteTool]

        def search_on_google(self, msg: SearchOnGoogleTool) -> str:
            print(f"SEARCHING ON GOOGLE: {msg.query}")
            self.expecting_tool_result = True
            self.expecting_tool_use = False
            return msg.handle()

        def open_website(self, msg: OpenWebsiteTool) -> str:
            print(f"OPENING WEBSITE: {msg.url}")
            self.expecting_tool_result = True
            self.expecting_tool_use = False
            return msg.handle()
