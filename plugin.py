# https://mwax911.medium.com/building-a-plugin-architecture-with-python-7b4ab39ad4fc

from typing import Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import langroid as lr
from langroid.agent.tools.orchestration import AgentDoneTool

from tools import QuestionTool, AnswerTool

# from model import Meta, Device


@dataclass
class Meta:
    name: str
    description: str
    version: str

    def __str__(self) -> str:
        return f"{self.name}: {self.version}"


class IPluginRegistry(type):
    plugin_registries: List[type] = list()

    def __init__(cls, name, bases, attrs):
        super().__init__(cls)
        if name != "PluginCore":
            IPluginRegistry.plugin_registries.append(cls)


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
    Plugin core class
    """

    meta: Optional[Meta]
    agent: Optional[PluginAgent]

    @abstractmethod
    def __init__(self) -> None:
        """
        Entry init block for plugins
        """

    # @abstractmethod
    # def invoke(self, **args):
    #     """
    #     Starts main plugin flow
    #     :param args: possible arguments for the plugin
    #     :return: a device for the plugin
    #     """
    #     pass
