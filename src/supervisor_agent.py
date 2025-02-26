# supervisor_agent.py
from typing import Literal, TypedDict, Annotated, List
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

# --- Định nghĩa mô hình định tuyến (Router) ---
class Router(BaseModel):
    # Nếu LLM định hướng cần truy vấn CSDL, trả về "databaseAgent" hoặc "FINISH" để không truy vấn
    routeToDB: Literal["databaseAgent", "FINISH"]
    # Nếu LLM định hướng cần phản hồi cho người dùng, trả về "userProxyAgent" hoặc "FINISH"
    routeToUser: Literal["userProxyAgent", "FINISH"]

class SupervisorAgent:
    def __init__(self, llm_api_key: str) -> None:
        self.llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash" ,
        temperature=0.5,
        google_api_key=llm_api_key
        )
        self.members = ["databaseAgent", "userProxyAgent"]
        self.system_prompt = (
            f"You are a Supervisor responsible for orchestrating a conversation among multiple specialized agents. This is your list of worker you can control {self.members} "
            "When a user request is received, your task is to analyze the request and determine which agent is best suited to handle it. "
            "Delegate the task accordingly, ensuring that the selected agent performs the necessary actions and returns a detailed response including its results and status. "
            "Once an agent completes its task, respond with 'FINISH' along with the agent's output for the user. "
            "When finished respond with FINISH."

        )

    def createSupervisorCommand(self, state: MessagesState) -> Command[Literal["databaseAgent", "userProxyAgent", "__end__"]]:
        """
        Hàm này sẽ dùng lịch sử chat hiện có để điều hành các agent nhân viên

        Args:
              state: là lịch sử chat trong phiên đó của người dùng

        Return:
              Command đến agent nhân viên mà supervisor kiểm soát

        """
        # Tạo tin nhắn hệ thống và kết hợp với lịch sử tin nhắn hiện có
        system_message = {"role": "system", "content": self.system_prompt}
        recent_messages = state["messages"][-3:]
        combined_messages = [system_message] + recent_messages
        print(combined_messages)

        router_response = self.llm.with_structured_output(Router).invoke(combined_messages)


        if router_response.routeToDB != "FINISH":
            destination = router_response.routeToDB
        elif router_response.routeToUser != "FINISH":
            destination = router_response.routeToUser
        else:
            destination = "__end__"

        return Command(goto=destination)


