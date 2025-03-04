import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



class DocumentAgent:
    def __init__(self, llm_api_key: str) -> None:
        """
        Xây dựng agent để truy vấn tài liệu:
          - Khởi tạo LLM Google Gemini
          - Tích hợp vector store từ VectorStoreManager
          - Trả lời câu hỏi của người dùng dựa trên tài liệu đã index

        Args:
            llm_api_key (str): API key cho LLM (Google Gemini).

        """

        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=llm_api_key)
        self.system_prompt = (
            "Bạn là một trợ lý thông minh chuyên về kiến thức toán học. "
            "Các tài liệu chứa kiến thức toán đã được nạp sẵn. "
            "Sử dụng các đoạn trích và thông tin liên quan từ các tài liệu đó để trả lời câu hỏi của người dùng một cách ngắn gọn, chính xác và súc tích (tối đa 3 câu). "
            "Nếu không tìm thấy thông tin phù hợp, hãy trả lời 'Tôi không biết'. "
            "Dưới đây là ngữ cảnh: {context}"
        )


    def query_document(self, query: str, vector_store: object):
        """Truy vấn tài liệu đã index."""
        if not vector_store:
            raise ValueError("Vector store chưa được khởi tạo. Hãy chạy initialize_vector_store trước.")

        retriever = vector_store.as_retriever()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        results = rag_chain.invoke({"input": query})
        return results
