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
# Chúng ta mở rộng MessagesState để lưu trữ cả lịch sử tin nhắn và danh sách các agent đã xử lý.
class State(MessagesState):
    agents: Annotated[list, add_messages]
    messages: Annotated[list, add_messages]



# --- Node 1: Supervisor (Định tuyến) ---
def supervisor_node(state: State) -> Command[Literal["databaseAgent", "userProxyAgent", "__end__"]]:
    # Khởi tạo SupervisorAgent với API key
    supervisor = SupervisorAgent(llm_api_key=SUPERVISOR_API_KEY)

    # Xác định agent nào cần xử lý trước
    supervisorCommand = supervisor.createSupervisorCommand(state)

    return supervisorCommand



# --- Node 2: Database (Xử lý truy vấn cơ sở dữ liệu) ---
def database_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    # Khởi tạo đối tượng DatabaseAgent với các thông số kết nối và API key
    db_agent = DatabaseAgent(
        db_user=DB_USER,
        db_password=DB_PASSWORD,
        db_host=DB_HOST,
        db_database=DB_NAME,
        llm_apiKey=DB_API_KEY
    )
    # Tạo truy vấn SQL dựa trên tin nhắn cuối cùng từ người dùng
    query = db_agent.llm_gen_sqlquery(state["messages"][-1].content)
    # Loại bỏ các mô tả không cần thiết để chỉ giữ lại câu lệnh SQL hợp lệ
    sqlCleaned = db_agent.clean_sql_query(query=query)
    # Thực thi truy vấn SQL và lấy kết quả
    sqlResponse = db_agent.execute_query(sqlCleaned)
    # Tạo phản hồi tự nhiên từ kết quả truy vấn, trả về dưới dạng AIMessage
    natural_response = db_agent.make_SQLresponse_natural(sqlResponse)

    # Lấy chuỗi nội dung từ AIMessage và xử lý marker kết thúc "<<END>>"
    response_str = natural_response.content.strip()
    marker_pattern = r'(<<END>>)$'
    is_end = bool(re.search(marker_pattern, response_str))
    # Loại bỏ marker khỏi nội dung phản hồi để hiển thị cho người dùng
    cleaned_message = re.sub(marker_pattern, '', response_str).strip()

    # Nếu phản hồi đánh dấu hoàn tất thì chuyển hướng đến trạng thái kết thúc, ngược lại quay lại supervisor
    goto_target = "__end__" if is_end else "supervisor"

    return Command(goto=goto_target, update={"messages": cleaned_message, "agents": ["databaseAgent"]})


# --- Node 3: User Proxy (Xử lý phản hồi cho người dùng) ---
def user_proxy_node(state: State) -> Command[Literal["supervisor"]]:
    user_agent = UserProxyAgent(name=AGENT_NAME, llm_api_key=USER_PROXY_API_KEY)
    # Nối toàn bộ nội dung tin nhắn trong lịch sử chat
    message_history = "\n".join([msg.content for msg in state["messages"]])
    # Giả sử user_agent trả về một danh sách các tin nhắn (scratchpad)
    user_response = user_agent.returnLLMRespose(state=message_history)
    return Command(goto="supervisor", update={"messages": user_response, "agents": ["userProxyAgent"]})




# --- Hàm chính chạy chương trình ---
def main():
    # Xây dựng đồ thị trạng thái với schema là SimpleState
    builder = StateGraph(state_schema=State)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("databaseAgent", database_node)
    builder.add_node("userProxyAgent", user_proxy_node)
    builder.add_edge(START, "supervisor")

    # Biên dịch đồ thị và gắn checkpointer vào để lưu trạng thái
    graph = builder.compile(checkpointer=memory)

    # Hiển thị sơ đồ đồ thị (Mermaid)
    #print(graph.get_graph().draw_mermaid())
    # Có thể vẽ sơ đồ PNG:
    #draw_mermaid_png(graph.get_graph().draw_mermaid(), output_file_path="diagram.png")

    # Vòng lặp chính nhận tin nhắn người dùng và xử lý qua đồ thị
    while True:
        user_input = input("Nhập tin nhắn (q để thoát): ")
        if user_input.lower() == "q":
            break

        # Gọi đồ thị với tin nhắn người dùng, config được truyền vào là tham số thứ 2
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        )
        # In ra tin nhắn cuối cùng được cập nhật từ các node
        for event in events:
            event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()