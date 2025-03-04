# user_proxy_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

class UserProxyAgent:
    """
    Đây là 1 module trả lời những câu hỏi bình thường của người dùng mà không liên quan đến thông tin trang web
    - Mục đích là hỗ trợ người dùng
    ví dụ như bạn là gì ?
    """
    def __init__(self, name: str, llm_api_key: str) -> None:
        self.name = name
        self.llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=llm_api_key
        )
        self.system_prompt = f"Bạn tên là {self.name}, một trợ lý ảo thông minh trên 1 trang web thuê gia sư, mục đích của bạn là giúp người dùng tương tác với các hệ thống và dịch vụ. Vai trò của bạn là nhận yêu cầu từ người dùng, xử lý thông tin, giao tiếp với các dịch vụ bên ngoài hoặc hệ thống nội bộ, sau đó phản hồi lại với kết quả phù hợp. Hãy đưa ra phản hồi bằng tiếng Việt cho câu sau"

    def returnLLMRespose(self, state: str) -> AIMessage:
        response = self.llm.invoke(self.system_prompt + state)
        return AIMessage(content=response.content)
