from typing_extensions import Annotated, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from supervisor_agent import SupervisorAgent
from database_agent import DatabaseAgent
from user_proxy_agent import UserProxyAgent
from document_rag import DocumentAgent
from langgraph.graph.message import add_messages
import os
import re

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from create_vector_store import VectorStoreManager
from IPython.display import display, Image
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
DOCUMENT_RAG_API_KEY = os.getenv("DOCUMENT_RAG_API_KEY")
AGENT_NAME = "Tutorium"
DOCUMENT_PATH = "C:/Users/admin/Desktop/Tutorium RAG Chatbot/data"

# --- Khởi tạo bộ lưu trữ (checkpointer) và cấu hình ---
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}


# --- Định nghĩa trạng thái (State) của đồ thị ---
class State(MessagesState):
    agents: Annotated[list, add_messages]  # Danh sách các agent tham gia
    messages: Annotated[list, add_messages]  # Danh sách tin nhắn trong hội thoại
    context_summary: str  # Tóm tắt ngữ cảnh hội thoại


# Cập nhật tóm tắt ngữ cảnh từ các tin nhắn gần nhất
def update_context_summary(state: dict) -> None:
    state.setdefault("context_summary", "")  # Đảm bảo key "context_summary" tồn tại
    last_messages = state.get("messages", [])[-5:]  # Lấy 5 tin nhắn cuối
    summary = "\n".join(
        msg.get("content", "") if isinstance(msg, dict) else msg.content
        for msg in last_messages
    )
    state["context_summary"] = summary  # Cập nhật tóm tắt vào state


# Load dữ liệu tài liệu để tạo vector space phục vụ RAG
def load_document_data_to_vector_space(document_folder_path: str) -> object:
    """
    Hàm này sẽ tạo ra vector space phục vụ cho RAG

    Args:
        document_folder_path: str đường link dẫn đến folder chứa document chính

    Returns:
        object: 1 object chứa vector space đọc từ các documents

    """
    vector_manager = VectorStoreManager(document_folder_path)
    vector_manager.load_documents_from_folder()
    vector_store = vector_manager.create_vector_store()
    return vector_store


# --- Node 1: Supervisor (Định tuyến) ---
def supervisor_node(state: State) -> Command[Literal["databaseAgent", "userProxyAgent", "__end__"]]:
    update_context_summary(state)
    supervisor = SupervisorAgent(llm_api_key=SUPERVISOR_API_KEY)
    supervisorCommand = supervisor.createSupervisorCommand(state)
    return supervisorCommand


# --- Node 2: Database (Xử lý truy vấn cơ sở dữ liệu) ---
def database_node(state: dict) -> Command[Literal["supervisor", "__end__"]]:
    update_context_summary(state)
    db_agent = DatabaseAgent(
        db_user=DB_USER,
        db_password=DB_PASSWORD,
        db_host=DB_HOST,
        db_database=DB_NAME,
        llm_apiKey=DB_API_KEY
    )
    # Tạo truy vấn từ tin nhắn cuối cùng và ngữ cảnh
    query_input = state["messages"][-1].content + "\nContext:\n" + state["context_summary"]
    query = db_agent.llm_gen_sqlquery(query_input)
    sqlCleaned = db_agent.clean_sql_query(query=query)
    sqlResponse = db_agent.execute_query(sqlCleaned)
    natural_response = db_agent.make_SQLresponse_natural(sqlResponse, userInput=state["messages"][-1].content)

    # Kiểm tra nếu có marker <<END>> để kết thúc
    response_str = natural_response.content.strip()
    marker_pattern = r'(<<END>>)$'
    is_end = bool(re.search(marker_pattern, response_str))
    cleaned_message = re.sub(marker_pattern, '', response_str).strip()
    goto_target = "__end__" if is_end else "supervisor"

    return Command(goto=goto_target, update={"messages": cleaned_message, "agents": ["databaseAgent"]})


# --- Node 3: User Proxy (Xử lý phản hồi cho người dùng) ---
def user_proxy_node(state: dict) -> Command[Literal["supervisor"]]:
    update_context_summary(state)
    user_agent = UserProxyAgent(name=AGENT_NAME, llm_api_key=USER_PROXY_API_KEY)
    message_history = "\n".join(
        msg.get("content", "") if isinstance(msg, dict) else msg.content
        for msg in state.get("messages", [])
    )
    full_context = message_history + "\nSummary:\n" + state["context_summary"]
    user_response = user_agent.returnLLMRespose(state=full_context)
    return Command(goto="supervisor", update={"messages": user_response, "agents": ["userProxyAgent"]})


# --- Node 4: Document RAG (Truy xuất thông tin từ tài liệu) ---
def document_rag_node(state: dict) -> Command[Literal["supervisor"]]:
    update_context_summary(state)
    da = DocumentAgent(llm_api_key=DOCUMENT_RAG_API_KEY)
    query_input = state["messages"][-1].content + "\nContext:\n" + state["context_summary"]
    user_response = da.query_document(query_input, vector_store)["answer"]
    return Command(goto="supervisor", update={"messages": user_response, "agents": ["documentRAG"]})


# --- Khởi tạo đồ thị xử lý ---
# Khởi tạo vector space phục vụ cho RAG
vector_store = load_document_data_to_vector_space(DOCUMENT_PATH)

builder = StateGraph(state_schema=State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("databaseAgent", database_node)
builder.add_node("userProxyAgent", user_proxy_node)
builder.add_node("documentRAG", document_rag_node)
builder.add_edge(START, "supervisor")
graph = builder.compile(checkpointer=memory)

print(graph.get_graph().draw_mermaid())
#draw_mermaid_png(graph.get_graph().draw_mermaid(), output_file_path="diagram.png")
# --- Hàm chính xử lý tin nhắn từ người dùng ---
def process_input_message(user_message: str) -> str:
    events = list(graph.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config,
        stream_mode="values",
    ))
    if events and events[-1].get("messages"):
        return events[-1]["messages"][-1].content
    return ""
