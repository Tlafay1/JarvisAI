from types import ModuleType
import langroid as lr
import typer
from rich.prompt import Prompt
import importlib
import pkgutil

from config import LLM_CONFIGS

from MainAgent import MainAgent
import plugins
import plugin
import inspect

app = typer.Typer()

lr.utils.logging.setup_colored_logging()


def iter_namespace(ns_pkg):
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


discovered_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg in iter_namespace(plugins)
}


def find_plugin_classes():
    agent_classes = []
    for name, module in discovered_plugins.items():
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, plugin.PluginCore):
                agent_classes.append(obj)
    return agent_classes


plugin_classes = find_plugin_classes()


@app.command()
def chat():
    main_agent = MainAgent(LLM_CONFIGS.get("small"))

    agents = []
    for agent_class in plugin_classes:
        agents.append(agent_class(LLM_CONFIGS.get("small")))

    main_agent.task.add_sub_task([agent.task for agent in agents])
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
