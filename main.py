import langroid as lr
import typer
import langroid.language_models as lm
from langroid.utils.configuration import settings
from BrowserAgent import BrowserAgent
from MainAgent import MainAgent
from rich.prompt import Prompt

CONTEXT_LENGTH = 128000

app = typer.Typer()

lr.utils.logging.setup_colored_logging()


@app.command()
def chat():
    settings.debug = False
    settings.cache = False
    llm_config = lm.OpenAIGPTConfig(
        chat_model="ollama/gemma2:27b",
        chat_context_length=CONTEXT_LENGTH,  # set this based on model
        max_output_tokens=1000,
        temperature=0.2,
        stream=True,
        timeout=45,
    )

    browser_agent = BrowserAgent(llm_config)
    main_agent = MainAgent(llm_config)
    adder_agent = lr.ChatAgent(lr.ChatAgentConfig(llm=llm_config))
    adder_task = lr.Task(
        adder_agent,
        name="adder_agent",
        system_message="""
        You are an expert on addition of numbers.
        When given numbers to add, simply return their sum, say nothing else
        """,
        single_round=True,
        use_tools=True,
        use_functions_api=False,
        enable_orchestration_tool_handling=True,
    )

    main_agent.task.add_sub_task([browser_agent.task, adder_task])
    main_agent.task.run("Ask the user for a task to execute")


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    tools: bool = typer.Option(
        False,
        "--tools",
        "-t",
        help="use langroid tools instead of OpenAI function-calling",
    ),
) -> None:
    lr.utils.configuration.set_global(
        lr.utils.configuration.Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat(tools)


if __name__ == "__main__":
    app()
