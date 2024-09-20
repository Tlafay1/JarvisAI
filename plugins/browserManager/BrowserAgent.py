# BrowserAgent.py
from typing import List, Optional
import langroid as lr
from langroid import ChatDocument
import langroid.language_models as lm
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.language_models.base import OpenAIToolCall

from tools import QuestionTool, AnswerTool, SearchOnGoogleTool, OpenWebsiteTool
from plugin import PluginCore, PluginAgent, Meta
from config import LLM_CONFIGS


class BrowserAgent(PluginCore):
    # meta = Meta(
    #     name="BrowserAgent",
    #     description="A plugin that allows the user to control the web browser.",
    #     version="0.1",
    # )

    def register_agents(self) -> List[PluginAgent] | None:
        config = lr.ChatAgentConfig(
            name="BrowserAgent",
            llm=LLM_CONFIGS.get("small"),
            system_message="""
                You are an expert on controlling the web browser.
                For ANY TASK you receive, you must use the appropriate tool to execute it.
                Once you the TASK is executed, you must send the result back to the user.
                EXTREMELY IMPORTANT: You must NOT execute the TASK yourself, use a tool ONLY.
                """,
        )
        return self.Agent(config)

    def register_tools(self) -> List[lr.ToolMessage] | None:
        return [SearchOnGoogleTool, OpenWebsiteTool]

    class Agent(PluginAgent, lr.ChatAgent):
        def init_state(self) -> None:
            self.current_query: str | None = None
            self.expecting_tool_result: bool = False
            self.expecting_tool_use = False

        def __init__(self, config: lr.ChatAgentConfig):
            super().__init__(config)
            self.config = config
            self.enable_message(
                [SearchOnGoogleTool, OpenWebsiteTool], use=True, handle=True
            )
            self.enable_message([QuestionTool, AnswerTool], use=False, handle=True)

        # def process_tool_results(
        #     self,
        #     results: str,
        #     id2result: OrderedDict[str, str] | None,
        #     tool_calls: List[OpenAIToolCall] | None = None,
        # ) -> Tuple[str, Dict[str, str] | None, str | None]:
        #     self.expecting_tool_result = False
        #     print(f"PROCESSING TOOL RESULTS CALLED")
        #     return super().process_tool_results(results, id2result, tool_calls)

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

        def handle_message_fallback(
            self, msg: str | ChatDocument
        ) -> str | ChatDocument | None:
            # we're here because msg has no tools
            if self.current_query is None:
                # did not receive a question tool, so short-circuit and return None
                return None
            if self.expecting_tool_use:
                return f"""
                You forgot to use a tool to execute the user query: {self.current_query}!!
                REMEMBER - you must ONLY execute the user's query based on
                a tool, and you MUST NOT EXECUTE them yourself.
                """

        def question_tool(self, msg: QuestionTool) -> str:
            self.current_query = msg.instruction
            return f"""
            User asked for this TASK to be executed: {msg.instruction}.
            Execute the TASK using the appropriate tool
            using the specified JSON format.
            """

        def answer_tool(self, msg: AnswerTool) -> AgentDoneTool:
            return AgentDoneTool(tools=[msg])

        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            if self.expecting_tool_result:
                current_query = self.current_query
                self.current_query = None
                self.expecting_tool_result = False
                result = super().llm_response_forget(message)
                answer = f"""
                Here are the actions executed based on the task: {current_query}
                ===
                {result}
                ===
                Decide whether you want to:
                - Execute other tasks, to fulfill the user's query, or
                - Present the final answer to the user.
                """
                ans_tool = AnswerTool(task_result=answer)
                print(f"ANSWER TOOL: {ans_tool}")
                return self.create_llm_response(tool_messages=[ans_tool])
            return super().llm_response_forget(message)
