from typing import List, Dict, Any, Union

import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader  # Các loader cho PDF, DOCX, TXT
from langchain_community.embeddings import HuggingFaceEmbeddings  # Sử dụng mô hình embedding của Hugging Face theo khuyến cáo mới
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI  # Vẫn sử dụng Google Gemini cho LLM
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*HuggingFaceEmbeddings.*deprecated.*")
warnings.filterwarnings("ignore", message=".*huggingface_hub.*symlinks.*")
import time
load_dotenv()

# --- Đọc biến môi trường ---
DB_API_KEY = os.getenv("DATABASE_GEMINI_API_KEY")


class DocumentAgent:
    def __init__(self, llm_api_key: str, document_folder_path: str) -> None:
        """
        Khởi tạo agent để xử lý tài liệu:
          - Đọc các tài liệu từ thư mục
          - Tách các đoạn văn (chunks)
          - Tạo vector store bằng Chroma thông qua HuggingFaceEmbeddings
          - Xây dựng retrieval chain kết hợp LLM để trả lời câu hỏi

        Args:
            llm_api_key (str): API key cho LLM (Google Gemini).
            document_folder_path (str): Đường dẫn thư mục chứa tài liệu.
        """
        self.llm_api_key = llm_api_key
        self.document_folder_path = document_folder_path
        self.supported_formats = ["pdf", "docx", "txt"]  # Các định dạng file được hỗ trợ
        self.text_chunks = []  # Danh sách các đoạn văn sau khi tách
        self.vector_store = None  # Vector store sẽ được tạo sau
        # Khởi tạo LLM chat với Google Gemini với model "gemini-1.5-flash"
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=self.llm_api_key)
        self.system_prompt = (
            "Bạn là một trợ lý thông minh chuyên về kiến thức toán học. "
            "Các tài liệu chứa kiến thức toán đã được nạp sẵn. "
            "Sử dụng các đoạn trích và thông tin liên quan từ các tài liệu đó để trả lời câu hỏi của người dùng một cách ngắn gọn, chính xác và súc tích (tối đa 3 câu). "
            "Nếu không tìm thấy thông tin phù hợp, hãy trả lời 'Tôi không biết'. "
            "Dưới đây là ngữ cảnh: {context}"
        )

    def load_and_process_document(self, file_path: str) -> List[Document]:
        """
        Tải tài liệu từ file và tách thành các đoạn văn nhỏ (chunks).

        Args:
            file_path (str): Đường dẫn đến file cần xử lý.

        Returns:
            list: Danh sách Document đã được tách thành các đoạn nhỏ.
        """
        file_extension = file_path.split(".")[-1].lower()  # Lấy phần mở rộng của file
        # Sử dụng các loader phù hợp với định dạng file
        if file_extension == "pdf":
            docs = PyMuPDFLoader(file_path).load()
        elif file_extension == "docx":
            docs = Docx2txtLoader(file_path).load()
        elif file_extension == "txt":
            docs = TextLoader(file_path).load()
        else:
            raise ValueError(f"Định dạng tệp không được hỗ trợ: {file_extension}")

        # Tách tài liệu thành các đoạn nhỏ bằng RecursiveCharacterTextSplitter
        # Với kích thước chunk là 1000 ký tự và chồng lấn 200 ký tự
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        return splits

    def load_documents_from_folder(self) -> List[Document]:
        """
        Duyệt qua thư mục được chỉ định, tải và xử lý tất cả các tài liệu hỗ trợ.

        Returns:
            list: Danh sách các đoạn văn đã được tách từ tất cả các tài liệu.
        """
        all_chunks = []
        # Lặp qua các file trong thư mục
        for filename in os.listdir(self.document_folder_path):
            file_path = os.path.join(self.document_folder_path, filename)
            # Nếu là file và định dạng file nằm trong danh sách hỗ trợ
            if os.path.isfile(file_path) and filename.split(".")[-1].lower() in self.supported_formats:

                splits = self.load_and_process_document(file_path)
                all_chunks.extend(splits)
        self.text_chunks = all_chunks  # Lưu lại các đoạn văn tách được
        return all_chunks

    def create_vector_store(self):
        if not self.text_chunks:
            raise ValueError("Chưa có text chunks. Vui lòng tải và xử lý tài liệu trước.")
        # Sử dụng HuggingFaceEmbeddings với mô hình chạy cục bộ
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = Chroma.from_documents(
            documents=self.text_chunks,
            embedding=hf_embeddings
        )
        return self.vector_store

    def query_document(self, query: str):
        retriever = self.vector_store.as_retriever()
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

# Ví dụ sử dụng DocumentAgent:
start_time = time.time()
da = DocumentAgent(llm_api_key=DB_API_KEY, document_folder_path="C:/Users/admin/Desktop/Agent/data")
da.load_documents_from_folder()
da.create_vector_store()
query = ""
print(query)
#print(da.query_document(query)["context"][0].page_content)
print(da.query_document(query)["answer"])

end_time = time.time()  # Kết thúc tính thời gian
print(f"Thời gian chạy của load_and_process_document: {end_time - start_time:.2f} giây")