import langroid as lr
import typer
from rich.prompt import Prompt
import torch


from config import LLM_CONFIGS

from MainAgent import MainAgent
from plugin import PluginManager

app = typer.Typer()

lr.utils.logging.setup_colored_logging()

plugin_manager = PluginManager()

torch.cuda.empty_cache()


@app.command()
def chat():
    main_agent = MainAgent(LLM_CONFIGS.get("small"))

    main_agent.task.add_sub_task(plugin_manager.tasks)
    question = Prompt.ask("What do you want to do ?")
    main_agent.task.run(question)


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
