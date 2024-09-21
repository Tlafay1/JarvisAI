import langroid as lr
from langroid import ChatDocument

from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.agent.tools.orchestration import PassTool

from typing import Optional
from config import LLM_CONFIGS
from plugin import PluginManager

from tools import QuestionTool, AnswerTool

plugin_manager = PluginManager()


class MainAgent:
    def __init__(self):
        self.agent = lr.ChatAgent(
            lr.ChatAgentConfig(
                llm=LLM_CONFIGS.get("small"),
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
                RecipientTool.create(
                    plugin_manager.plugin_names,
                ),
                QuestionTool,
                PassTool,
            ]
        )
        self.agent.enable_message(
            plugin_manager.tools,
            use=False,
            handle=True,
        )
        self.task = lr.Task(
            self.agent,
        )
        self.task.add_sub_task(plugin_manager.tasks)

    def run(self, message: str) -> ChatDocument:
        return self.task.run(message)

    class Agent(lr.ChatAgent):
        def init_state(self) -> None:
            super().init_state()
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

                REMINDER: If you still need to execute a task, then use the `question_tool`
                to execute a SINGLE task that can be executed by an agent.
                """
            elif self.expecting_question_tool:
                return f"""
                You must give an instruction using the `question_tool` in the specified format,
                to break down the user's original query: {self.original_query} into
                smaller instructions that can be executed by a specialist.
                """

        def question_tool(self, tool: QuestionTool) -> str | PassTool:
            self.expecting_task_answer = True
            self.expecting_question_tool = False
            return PassTool()

        def answer_tool(self, tool: AnswerTool) -> str:
            if not self.expecting_question:
                return ""

            self.expecting_question = False
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
