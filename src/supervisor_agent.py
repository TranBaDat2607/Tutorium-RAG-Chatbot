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
    # Nếu LLM định hướng cần lấy câu hỏi hay kiến thức trong vector store thì trả về "documentRAG" hoặc "FINISH"
    routeToRAG: Literal["documentRAG", "FINISH"]


class SupervisorAgent:
    def __init__(self, llm_api_key: str) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.5,
            google_api_key=llm_api_key
        )
        self.members = ["databaseAgent", "documentRAG", "userProxyAgent"]
        # Cải tiến system prompt:
        # 1. Bổ sung Few-Shot Examples (ví dụ minh họa)
        # 2. Thêm chỉ dẫn Chain-of-Thought (suy nghĩ từng bước)
        # 3. Nâng cao ngữ cảnh và chỉ dẫn cụ thể
        self.system_prompt = (
            f"You are a Supervisor responsible for orchestrating a conversation among multiple specialized agents. "
            f"Your available worker agents are: {self.members}. "
            "When a user request is received, analyze it thoroughly and decide which agent is best suited to handle it. "
            "Think step-by-step about the decision process before delegating. "
            "Here are few-shot examples to guide your decision:\n"
            "Example 1:\n"
            "User request: 'Tìm kiếm thông tin người dùng theo username'.\n"
            "Decision: Delegate to 'databaseAgent'.\n"
            "Example 2:\n"
            "User request: 'hãy giải thích cho tôi khái niệm sau'.\n"
            "Decision: Delegate to 'documentRAG' take some question then ask them for clarify their level.\n"
            "Example 3:\n"
            "User request: 'Xin vui lòng trả lời câu hỏi của tôi về sản phẩm'.\n"
            "Decision: Delegate to 'userProxyAgent'.\n\n"
            "After delegating, ensure that the selected agent performs the necessary actions and returns a detailed response including its results and status. "
            "Once an agent completes its task, respond with 'FINISH' along with the agent's output for the user. "
            "When all tasks are complete, simply respond with 'FINISH'."
        )

    def createSupervisorCommand(self, state: dict) -> Command[Literal["databaseAgent", "userProxyAgent", "__end__"]]:
        """
        Sử dụng lịch sử chat hiện có để đưa ra quyết định điều phối các agent.

        Args:
            state: Lịch sử chat của phiên tương tác.

        Returns:
            Command: Lệnh chuyển hướng đến agent phù hợp.
        """
        # Tăng cường ngữ cảnh: sử dụng nhiều tin nhắn gần đây hơn (ví dụ 5 tin nhắn)
        system_message = {"role": "system", "content": self.system_prompt}
        recent_messages = state["messages"][-5:]
        combined_messages = [system_message] + recent_messages

        router_response = self.llm.with_structured_output(Router).invoke(combined_messages)

        if router_response.routeToDB != "FINISH":
            destination = router_response.routeToDB
        elif router_response.routeToUser != "FINISH":
            destination = router_response.routeToUser
        elif router_response.routeToRAG != "FINISH":
            destination = router_response.routeToRAG
        else:
            destination = "__end__"
        print(f"Routing to: {destination}")
        return Command(goto=destination)
