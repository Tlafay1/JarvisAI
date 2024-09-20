from typing import List

from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import Field
from langroid.agent.tools.orchestration import SendTool

import webbrowser


class QuestionTool(ToolMessage):
    request: str = "question_tool"
    purpose: str = "Give a SINGLE <instruction> that can be executed by someone else."
    instruction: str

    @classmethod
    def examples(cls) -> List[ToolMessage]:
        return [
            cls(instruction="Open Stack Overflow"),
            cls(instruction="Search for 'how to make a cake'"),
            cls(instruction="Find the best restaurant in New York"),
            cls(instruction="Add 2 + 2"),
            cls(instruction="Send an email to John"),
            cls(instruction="Translate 'hello' to French"),
            cls(instruction="What is RecipientTool in Langroid?"),
        ]


class AnswerTool(ToolMessage):
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


class LangroidDocumentationSearchTool(ToolMessage):
    request = "langroid_doc_search"
    purpose = "To search the Langroid documentation for <query>."
    query: str = Field(..., description="Query to search in Langroid documentation")

    def handle(self) -> str:
        webbrowser.open_new_tab(f"https://langroid.org/docs/search?q={self.query}")
        return SendTool(
            to="User",
            content=f"Opened Langroid documentation search for '{self.query}'",
        )
