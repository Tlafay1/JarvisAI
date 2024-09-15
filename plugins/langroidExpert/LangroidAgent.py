import typer
import langroid as lr
import langroid.language_models as lm
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from rich.prompt import Prompt

from config import LLM_CONFIGS

from tools import QuestionTool, AnswerTool
from plugin import PluginCore

app = typer.Typer()


class LangroidAgent(PluginCore):
    def __init__(self, llm_config: lm.OpenAIGPTConfig):
        config = DocChatAgentConfig(
            name="LangroidAgent",
            llm=llm_config,
            doc_paths=[
                "./langroid-source.md",
                "./langroid-examples.md",
            ],
            system_message="""
                You are an expert about the Langroid LLM framework.
                Answer my question about docs.
                EXTREMELY IMPORTANT: If you don't know the answer, EXPLICITELY SAY you don't know.
                """,
        )

        self.agent = DocChatAgent(config)
        self.agent.enable_message([QuestionTool, AnswerTool], use=False, handle=True)
        self.task = lr.Task(
            self.agent, single_round=False, interactive=False, llm_delegate=True
        )

    def question_tool(self, msg: QuestionTool) -> str:
        print(f"User asked this question: {msg.question}")
        self.curr_query = msg.question
        self.expecting_search_tool = True
        return f"""
        User asked this question: {msg.question}.
        Answer the question using the docs.
        """


if __name__ == "__main__":

    @app.command()
    def main():
        langroid_agent = LangroidAgent(llm_config=LLM_CONFIGS.get("medium"))
        question = Prompt.ask("What do you want to know ?")

        q_doc = langroid_agent.task.agent.create_agent_response(
            tool_messages=[QuestionTool(instruction=question)]
        )

        result = langroid_agent.task.run(question)

        print(result)

        # tools = langroid_agent.task.agent.get_tool_messages(result)
        # assert len(tools) == 1
        # assert isinstance(tools[0], AnswerTool)

    app()
