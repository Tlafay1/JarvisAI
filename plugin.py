# https://mwax911.medium.com/building-a-plugin-architecture-with-python-7b4ab39ad4fc

from typing import Optional, List, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
import langroid as lr
from langroid.agent.tools.orchestration import AgentDoneTool
import importlib
import pkgutil

import plugins
import inspect
from tools import QuestionTool, AnswerTool


@dataclass
class Meta:
    name: str
    description: str
    version: str

    def __str__(self) -> str:
        return f"{self.name}: {self.version}"


class PluginAgent(ABC):
    @abstractmethod
    def init_state(self) -> None:
        pass

    @abstractmethod
    def handle_message_fallback(
        self, msg: str | lr.ChatDocument
    ) -> str | lr.ChatDocument | None:
        pass

    @abstractmethod
    def question_tool(self, msg: QuestionTool) -> str:
        pass

    @abstractmethod
    def answer_tool(self, msg: AnswerTool) -> AgentDoneTool:
        pass


class PluginCore(ABC):
    """
    PluginCore is the base class for all plugins.

    It provides the basic structure for a plugin, including the registration of agents, tools, and tasks.
    """

    agents: Optional[PluginAgent | List[PluginAgent]] = None
    tools: Optional[lr.ToolMessage | List[lr.ToolMessage]] = None
    meta: Optional[Meta]

    def register_agents(self) -> PluginAgent | List[PluginAgent] | None:
        """
        Registers the agents for the plugin.

        Returns:
            List[PluginAgent] | None: A list of PluginAgent instances if agents are registered,
                                       otherwise None.
        """
        return self.agents

    def register_tools(self) -> lr.ToolMessage | List[lr.ToolMessage] | None:
        """
        Registers and returns a list of tools.

        Returns:
            List[lr.ToolMessage] | None: A list of ToolMessage objects if tools are available, otherwise None.
        """
        return self.tools

    def register_tasks(self) -> lr.Task | List[lr.Task] | None:
        """
        Registers tasks for each agent in the `self.agents` list.
        Note that you very likely don't want to override this method,
        unless you know what you're doing.

        Returns:
            lr.Task | None: A list of `lr.Task` objects for each agent, or None if no agents are present.
        """
        agents = [self.register_agents()]
        return (
            [
                lr.Task(agent, single_round=False, interactive=False, llm_delegate=True)
                for agent in agents
            ]
            if agents
            else []
        )


class PluginManager:
    """
    PluginManager is responsible for managing plugins, agents, tools, and tasks.
    """

    __plugins: List[PluginCore] = []
    __agents: List[PluginAgent] = []
    __tools: List[lr.ToolMessage] = []
    __tasks: List[lr.Task] = []

    def __init__(self):
        self.load_plugins()

    def __iter_namespace(self, ns_pkg):
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    def __find_plugin_classes(self, discovered_plugins: dict) -> List[Type[PluginCore]]:
        agent_classes: List[Type[PluginCore]] = []
        for _, module in discovered_plugins.items():
            for _, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, PluginCore):
                    agent_classes.append(obj)
        return agent_classes

    @property
    def agents(self) -> List[PluginAgent]:
        return self.__agents

    @property
    def tools(self) -> List[lr.ToolMessage]:
        return self.__tools

    @property
    def tasks(self) -> List[lr.Task]:
        return self.__tasks

    def load_plugins(self) -> None:
        discovered_plugins = {
            name: importlib.import_module(name)
            for _, name, _ in self.__iter_namespace(plugins)
        }
        plugins_classes = self.__find_plugin_classes(discovered_plugins)
        self.__plugins = [plugin() for plugin in plugins_classes]
        self.register_agents()
        self.register_tools()
        self.register_tasks()

    def reload_plugins(self) -> None:
        self.__plugins = list()
        self.load_plugins()

    def register_agents(self) -> None:
        for plugin in self.__plugins:
            self.__agents.append(plugin.register_agents())

    def register_tools(self) -> None:
        for plugin in self.__plugins:
            self.__tools.append(plugin.register_tools())

    def register_tasks(self) -> None:
        for plugin in self.__plugins:
            self.__tasks.append(plugin.register_tasks())
