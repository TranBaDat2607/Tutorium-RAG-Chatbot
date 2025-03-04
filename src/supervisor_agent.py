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
        model="gemini-1.5-flash" ,
        temperature=0.5,
        google_api_key=llm_api_key
        )
        self.members = ["databaseAgent", "documentRAG", "userProxyAgent"]
        self.system_prompt = (
            f"You are a Supervisor responsible for orchestrating a conversation among multiple specialized agents. "
            f"Your list of worker agents: {self.members}. Your task is to analyze user requests and assign them to the most suitable agent, ensuring an efficient and accurate response. "

            "### Agent Responsibilities:\n"
            "- **databaseAgent**: Responsible for retrieving structured data from the database, such as teacher information, class schedules, and other academic-related records.\n"
            "- **documentRAG**: Specialized in answering high school-level mathematics questions. It provides problem-solving steps and final answers.\n"
            "- **userProxyAgent**: Handles general inquiries, greetings, or any request that does not fit the above categories.\n"

            "### Supervisor Instructions:\n"
            "1. **Analyze the User's Request**:\n"
            "   - If it asks for specific data related to teachers, schedules, or structured academic records → Assign to `databaseAgent`.\n"
            "   - If it is a high school mathematics question requiring problem-solving → Assign to `documentRAG`.\n"
            "   - If it is a general question (greetings, unrelated inquiries, or out-of-scope requests) → Assign to `userProxyAgent`.\n"

            "2. **Delegate the Task**:\n"
            "   - Direct the chosen agent to process the request.\n"
            "   - Ensure the agent provides a structured response with results and status.\n"

            "3. **Finalize the Response**:\n"
            "   - Once an agent completes its task, return its output using the format: `FINISH: <agent_name>: <response>`.\n"
            "   - If the request is unclear, ask the user for clarification before assigning an agent.\n"

            "Your goal is to ensure smooth task delegation and return precise, well-structured responses. "
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

        router_response = self.llm.with_structured_output(Router).invoke(combined_messages)


        if router_response.routeToDB != "FINISH":
            destination = router_response.routeToDB
        elif router_response.routeToUser != "FINISH":
            destination = router_response.routeToUser
        elif router_response.routeToRAG != "FINISH":
            destination = router_response.routeToRAG
        else:
            destination = "__end__"
        print(destination)

        return Command(goto=destination)


