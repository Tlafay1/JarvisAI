import langroid as lr
import langroid.language_models as lm
from langroid import ChatDocument

from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.agent.tools.orchestration import PassTool, ForwardTool

from typing import Optional

from BrowserAgent import SearchOnGoogleTool, OpenWebsiteTool
from tools import QuestionTool


class MainAgent:
    def __init__(self, llm_config: lm.OpenAIGPTConfig):
        self.agent = lr.ChatAgent(
            lr.ChatAgentConfig(
                llm=llm_config,
                system_message="""
                You are Jarvis, a resourceful AI assistant, able to think step by step to execute
                complex TASKS from the user. You must break down complex TASKS into
                simpler TASKS that can be executed by a specialist. You must send me
                (the user) each TASK ONE BY ONE, using the `question_tool` in
                the specified format, and I will execute the TASK and send you
                a brief answer.
                VERY IMPORTANT: You can not execute the TASK yourself, use a tool ONLY.
                """,
            )
        )
        self.agent.enable_message(
            [
                RecipientTool.create(["BrowserAgent"]),
                QuestionTool,
                PassTool,
                # ForwardTool,
            ]
        )
        self.agent.enable_message(
            [SearchOnGoogleTool, OpenWebsiteTool, QuestionTool],
            use=False,
            handle=True,
        )
        self.task = lr.Task(
            self.agent,
        )

    class Agent(lr.ChatAgent):
        def init_state(self) -> None:
            self.expecting_question_tool = False

        def user_response(
            self,
            msg: Optional[str | ChatDocument] = None,
        ) -> Optional[ChatDocument]:
            self.expecting_question_tool = False
            return super().user_response(msg)

        def question_tool(self, tool: QuestionTool) -> str | PassTool:
            self.expecting_question_tool = False
            return PassTool()

        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            if self.original_query is None:
                self.original_query = (
                    message if isinstance(message, str) else message.content
                )
                # just received user query, so we expect a question tool next
                self.expecting_question_tool = True
            if self.expecting_question_tool:
                return super().llm_response(message)

        def handle_message_fallback(
            self, msg: str | ChatDocument
        ) -> str | ChatDocument | None:
            if self.expecting_question_tool:
                return f"""
                You must execute a TASK using the `question_tool` in the specified format,
                to break down the user's original query: {self.original_query} into
                smaller tasks that can be executed by a specialist.
                """
