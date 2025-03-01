from typing_extensions import Annotated, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from supervisor_agent import SupervisorAgent
from database_agent import DatabaseAgent
from user_proxy_agent import UserProxyAgent
from langgraph.graph.message import add_messages
import os
import re

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langchain_core.runnables.graph_mermaid import draw_mermaid_png

from dotenv import load_dotenv

load_dotenv()

# --- Đọc biến môi trường cho cơ sở dữ liệu và API key ---
DB_USER = os.getenv("MYSQL_USER")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD")
DB_HOST = os.getenv("MYSQL_HOST")
DB_NAME = os.getenv("MYSQL_DATABASE")
DB_API_KEY = os.getenv("DATABASE_GEMINI_API_KEY")
SUPERVISOR_API_KEY = os.getenv("SUPERVISOR_GEMINI_API_KEY")
USER_PROXY_API_KEY = os.getenv("USERPROXY_GEMINI_API_KEY")
AGENT_NAME = "Tutorium"


# --- Khởi tạo bộ lưu trữ (checkpointer) và cấu hình ---
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}


# --- Định nghĩa trạng thái (State) của đồ thị ---
class State(MessagesState):
    agents: Annotated[list, add_messages]
    messages: Annotated[list, add_messages]
    context_summary: str

def update_context_summary(state: dict) -> None:
    # Đảm bảo rằng state có key "context_summary"
    state.setdefault("context_summary", "")
    # Lấy 5 tin nhắn cuối, nếu có
    last_messages = state.get("messages", [])[-5:]
    # Tạo tóm tắt từ các tin nhắn (giả sử mỗi tin nhắn là dict có key "content")
    summary = "\n".join(
        msg.get("content", "") if isinstance(msg, dict) else msg.content
        for msg in last_messages
    )
    # Cập nhật lại context_summary trong state
    state["context_summary"] = summary


# --- Node 1: Supervisor (Định tuyến) ---
def supervisor_node(state: State) -> Command[Literal["databaseAgent", "userProxyAgent", "__end__"]]:
    update_context_summary(state)  # Cập nhật tóm tắt context
    supervisor = SupervisorAgent(llm_api_key=SUPERVISOR_API_KEY)
    # Ví dụ: truyền tóm tắt context cho LLM nếu cần
    supervisorCommand = supervisor.createSupervisorCommand(state)
    return supervisorCommand



# --- Node 2: Database (Xử lý truy vấn cơ sở dữ liệu) ---
def database_node(state: dict) -> Command[Literal["supervisor", "__end__"]]:
    update_context_summary(state)  # Cập nhật tóm tắt context
    db_agent = DatabaseAgent(
        db_user=DB_USER,
        db_password=DB_PASSWORD,
        db_host=DB_HOST,
        db_database=DB_NAME,
        llm_apiKey=DB_API_KEY
    )
    # Sử dụng cú pháp truy cập key của dictionary cho "context_summary"
    query_input = state["messages"][-1].content + "\nContext:\n" + state["context_summary"]
    query = db_agent.llm_gen_sqlquery(query_input)
    sqlCleaned = db_agent.clean_sql_query(query=query)
    sqlResponse = db_agent.execute_query(sqlCleaned)
    natural_response = db_agent.make_SQLresponse_natural(sqlResponse, userInput=state["messages"][-1].content)

    response_str = natural_response.content.strip()
    marker_pattern = r'(<<END>>)$'
    is_end = bool(re.search(marker_pattern, response_str))
    cleaned_message = re.sub(marker_pattern, '', response_str).strip()

    goto_target = "__end__" if is_end else "supervisor"

    return Command(goto=goto_target, update={"messages": cleaned_message, "agents": ["databaseAgent"]})




# --- Node 3: User Proxy (Xử lý phản hồi cho người dùng) ---
def user_proxy_node(state: dict) -> Command[Literal["supervisor"]]:
    update_context_summary(state)  # Cập nhật tóm tắt context
    user_agent = UserProxyAgent(name=AGENT_NAME, llm_api_key=USER_PROXY_API_KEY)
    # Nối lịch sử tin nhắn. Giả sử mỗi tin nhắn là một dict có key "content" hoặc đối tượng có thuộc tính content.
    message_history = "\n".join(
        msg.get("content", "") if isinstance(msg, dict) else msg.content
        for msg in state.get("messages", [])
    )
    # Sử dụng cú pháp dictionary để truy cập tóm tắt context
    full_context = message_history + "\nSummary:\n" + state["context_summary"]
    user_response = user_agent.returnLLMRespose(state=full_context)
    return Command(goto="supervisor", update={"messages": user_response, "agents": ["userProxyAgent"]})




# --- Hàm chính chạy chương trình ---
builder = StateGraph(state_schema=State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("databaseAgent", database_node)
builder.add_node("userProxyAgent", user_proxy_node)
builder.add_edge(START, "supervisor")
graph = builder.compile(checkpointer=memory)

def process_input_message(user_message: str) -> str:
    """
    Hàm nhận tin nhắn từ người dùng, xử lý qua đồ thị và trả về phần content của tin nhắn cuối cùng.
    """
    # Lấy danh sách các event (chuyển sang list để có thể lấy phần tử cuối cùng)
    events = list(graph.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config,
        stream_mode="values",
    ))
    if events and events[-1].get("messages"):
        # Giả sử mỗi message có thuộc tính "content"
        return events[-1]["messages"][-1].content
    return ""