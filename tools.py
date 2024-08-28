from typing import List

import langroid as lr
from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import Field
from langroid.agent.tools.orchestration import SendTool

import webbrowser


class QuestionTool(lr.ToolMessage):
    request: str = "question_tool"
    purpose: str = "Return a SINGLE <instruction> that can be executed by someone else."
    question: str

    @classmethod
    def examples(cls) -> List[lr.ToolMessage]:
        return [
            cls(question="Open Stack Overflow"),
            cls(question="Search for 'how to make a cake'"),
            cls(question="Find the best restaurant in New York"),
            cls(question="Add 2 + 2"),
            cls(question="Send an email to John"),
        ]


class AnswerTool(lr.ToolMessage):
    request = "answer_tool"
    purpose = "Present the <task_result> from a TASK execution"
    task_result: str


class OpenWebsiteTool(ToolMessage):
    request = "open_website"
    purpose = "To open <url> in a new browser tab."
    url: str = Field(..., description="URL of the website to open")

    def handle(self) -> str:
        webbrowser.open_new_tab(self.url)
        return SendTool(to="User", content=f"Opened website '{self.url}'")


class SearchOnGoogleTool(ToolMessage):
    request = "search_on_google"
    purpose = "To perform a Google search based on <query>."
    query: str = Field(..., description="Query to search on Google")

    def handle(self) -> str:
        webbrowser.open_new_tab(f"https://www.google.com/search?q={self.query}")
        return SendTool(to="User", content=f"Opened Google search for '{self.query}'")
