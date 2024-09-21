# https://mwax911.medium.com/building-a-plugin-architecture-with-python-7b4ab39ad4fc

from typing import Iterable, Optional, List, Type
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


class PluginAgent(lr.ChatAgent, ABC):
    def init_state(self) -> None:
        super().init_state()
        self.current_query: str | None = None
        self.expecting_tool_result: bool = False
        self.expecting_tool_use = False

    def __init__(self, config: lr.ChatAgentConfig):
        super().__init__(config)
        self.config = config
        self.enable_message(self.register_tools())
        self.enable_message([QuestionTool, AnswerTool], use=False, handle=True)

    @abstractmethod
    def register_tools(self) -> List[lr.ToolMessage] | None:
        return None

    def handle_message_fallback(
        self, msg: str | lr.ChatDocument
    ) -> str | lr.ChatDocument | None:
        print("FALLBACK")
        if self.current_query is None:
            return None
        if self.expecting_tool_use:
            return f"""
                You forgot to use a tool to execute the user query: {self.current_query}!!
                REMEMBER - you must ONLY execute the user's query based on
                a tool, and you MUST NOT EXECUTE them yourself.
                """

    def question_tool(self, msg: QuestionTool) -> str:
        print("QUESTION TOOL")
        self.current_query = msg.instruction
        self.expecting_tool_use = True
        return f"""
        User asked for this TASK to be executed: {msg.instruction}.
        Execute the TASK using the appropriate tool
        using the specified JSON format.
        """

    def answer_tool(self, msg: AnswerTool) -> AgentDoneTool:
        print("ANSWER TOOL")
        return AgentDoneTool(tools=[msg])

    def llm_response(self, msg: str | lr.ChatDocument) -> str | lr.ChatDocument | None:
        print("LLM RESPONSE: ", msg)
        if self.expecting_tool_result:
            current_query = self.current_query
            self.current_query = None
            self.expecting_tool_result = False
            self.expecting_tool_use = False
            result = super().llm_response_forget(msg)

            answer = f"""
            Here is the result of the execution of the task: {current_query}.
            ===
            {result}
            ===
            """
            answer_tool = AnswerTool(task_result=answer)
            return self.create_llm_response(tool_messages=[answer_tool])
        result = super().llm_response_forget(msg)
        return result


class PluginCore(ABC):
    """
    PluginCore is the base class for all plugins.

    It provides the basic structure for a plugin, including the registration of agents, tools, and tasks.
    """

    agents: Optional[PluginAgent | List[PluginAgent]] = None
    tools: Optional[lr.ToolMessage | List[lr.ToolMessage]] = None

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

    @property
    def plugin_names(self) -> List[str]:
        return [plugin.Meta.name for plugin in self.__plugins]

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
            agents = plugin.register_agents()
            if not agents:
                continue
            if isinstance(agents, Iterable):
                self.__agents.extend(agents)
            else:
                self.__agents.append(agents)

    def register_tools(self) -> None:
        for plugin in self.__plugins:
            tools = plugin.register_tools()
            if not tools:
                continue
            if isinstance(tools, Iterable):
                self.__tools.extend(tools)
            else:
                self.__tools.append(tools)

    def register_tasks(self) -> None:
        for plugin in self.__plugins:
            tasks = plugin.register_tasks()
            if not tasks:
                continue
            if isinstance(tasks, Iterable):
                self.__tasks.extend(tasks)
            else:
                self.__tasks.append(tasks)
