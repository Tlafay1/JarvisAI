import langroid as lr
import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.agent.tools.orchestration import SendTool

from typing import Optional

from BrowserAgent import SearchOnGoogleTool, OpenWebsiteTool


class MainAgent:
    def __init__(self, llm_config: lm.OpenAIGPTConfig):
        self.agent = lr.ChatAgent(
            lr.ChatAgentConfig(
                llm=llm_config,
                system_message="""
                You are Jarvis, a helpful AI assistant.

                You will receive one or multiple INSTRUCTIONS from me (the user).
                Your goal is to execute the INSTRUCTION given.
                However you do not know how to perform those INSTRUCTIONS.
                You can take the help of as many people as you need to perform the
                INSTRUCTION. You can ask for help from the same person multiple times.
                You will send the instruction to the person who is best suited to fulfill it.

                IMPORTANT: Only send ONE INSTRUCTION at a time.

                You must clearly specify who you are sending the instruction to, using the
                `recipient_message` tool/function-call, where the `content` field
                is the instruction you want to send, and the `recipient` field is the name
                of the intended recipient, either "browser_agent" or "adder_agent".

                Once all INSTRUCTIONS have been executed, say DONE and show me the result.
                Start by greeting the user and asking them what they need help with.
                """,
            )
        )
        self.agent.enable_message([RecipientTool])
        self.agent.enable_message(
            [SearchOnGoogleTool, OpenWebsiteTool],
            use=False,
            handle=True,
        )
        self.task = lr.Task(
            self.agent,
        )

    class Agent(lr.ChatAgent):
        def init_state(self) -> None:
            self.llm_responded = False

        def user_response(
            self,
            msg: Optional[str | ChatDocument] = None,
        ) -> Optional[ChatDocument]:
            self.llm_responded = False
            return super().user_response(msg)

        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            self.llm_responded = True
            return super().llm_response(message)

        def handle_message_fallback(
            self, msg: str | ChatDocument
        ) -> str | ChatDocument | lr.ToolMessage | None:
            if self.llm_responded:
                self.llm_responded = False
                # LLM generated non-tool msg => send to user
                content = msg.content if isinstance(msg, ChatDocument) else msg
                return SendTool(to="User", content=content)
