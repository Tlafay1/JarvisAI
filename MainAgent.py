import langroid as lr
import langroid.language_models as lm
from langroid import ChatDocument

from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.agent.tools.orchestration import PassTool

from typing import Optional

from tools import QuestionTool, AnswerTool, SearchOnGoogleTool, OpenWebsiteTool


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
                RecipientTool.create(["BrowserAgent", "LangroidAgent"]),
                QuestionTool,
                PassTool,
            ]
        )
        self.agent.enable_message(
            [SearchOnGoogleTool, OpenWebsiteTool, AnswerTool],
            use=False,
            handle=True,
        )
        self.task = lr.Task(
            self.agent,
        )

    class Agent(lr.ChatAgent):
        def init_state(self) -> None:
            self.expecting_question_tool = False
            self.expecting_question: bool = False
            self.expecting_task_answer = False
            self.original_query: str | None = None

        def handle_message_fallback(
            self, msg: str | ChatDocument
        ) -> str | ChatDocument | None:
            if self.expecting_question:
                return """
                You may have intended to use a tool, but your JSON format may be wrong.

                REMINDER: If you still need to ask a question, then use the `question_tool`
                to ask a SINGLE question that can be answered from a web search.
                """
            elif self.expecting_question_tool:
                return f"""
                You must give an instruction using the `question_tool` in the specified format,
                to break down the user's original query: {self.original_query} into
                smaller instructions that can be executed by a specialist.
                """

        def user_response(
            self,
            msg: Optional[str | ChatDocument] = None,
        ) -> Optional[ChatDocument]:
            self.expecting_question_tool = False
            return super().user_response(msg)

        def question_tool(self, tool: QuestionTool) -> str | PassTool:
            self.expecting_task_answer = True
            self.expecting_question_tool = False
            return PassTool()

        def answer_tool(self, tool: AnswerTool) -> str:
            self.expecting_question = True
            self.expecting_task_answer = False
            return f"""
            Here's the result of the task execution: {tool.task_result}.
            Now decide whether you want to:
            - return the result of the task's execution to the user, OR
            - execute another task using the `question_tool`
                (Maybe REPHRASE the task to get MORE ACCURATE execution).
            """

        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            if self.original_query is None:
                self.original_query = (
                    message if isinstance(message, str) else message.content
                )
                # just received user query, so we expect a question tool next
                self.expecting_question_tool = True
            if self.expecting_question_tool or self.expecting_question:
                return super().llm_response(message)
