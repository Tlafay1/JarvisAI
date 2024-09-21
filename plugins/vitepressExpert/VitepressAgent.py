from typing import List
import typer
import langroid as lr
import langroid.language_models as lm
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from rich.prompt import Prompt

from config import LLM_CONFIGS
from plugin import PluginAgent, PluginCore

from tools import QuestionTool, AnswerTool

app = typer.Typer()


class VitepressAgent(PluginCore):
    class Meta:
        name = "VitepressAgent"
        description = "A plugin that allows the user to ask questions about the Vitepress framework."
        version = "0.1"

    def register_agents(self) -> PluginAgent | List[PluginAgent] | None:
        config = DocChatAgentConfig(
            name=self.Meta.name,
            llm=LLM_CONFIGS.get("small"),
            doc_paths=[
                "./vitepress-source.md",
                "./vitepress-docs.md",
            ],
            system_message="""
                You are an expert about the Vitepress framework.
                Answer my question about docs.
                EXTREMELY IMPORTANT: If you don't know the answer, EXPLICITELY SAY you don't know.
                """,
        )

        agent = DocChatAgent(config)
        agent.enable_message([QuestionTool, AnswerTool], use=False, handle=True)
        return agent


if __name__ == "__main__":

    @app.command()
    def main():
        langroid_agent = VitepressAgent(llm_config=LLM_CONFIGS.get("medium"))
        question = Prompt.ask("What do you want to know ?")

        # q_doc = langroid_agent.task.agent.create_agent_response(
        #     tool_messages=[QuestionTool(instruction=question)]
        # )

        result = langroid_agent.task.run(question)

        print(result)

        # tools = langroid_agent.task.agent.get_tool_messages(result)
        # assert len(tools) == 1
        # assert isinstance(tools[0], AnswerTool)

    app()
