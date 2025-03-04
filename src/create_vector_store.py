import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class VectorStoreManager:
    def __init__(self, document_folder_path: str) -> None:
        """
        Quản lý vector store:
          - Đọc tài liệu từ thư mục
          - Tách văn bản thành các đoạn nhỏ (chunks)
          - Tạo vector store bằng Chroma và HuggingFaceEmbeddings

        Args:
            document_folder_path (str): Đường dẫn thư mục chứa tài liệu.
        """
        self.document_folder_path = document_folder_path
        self.supported_formats = ["pdf", "docx", "txt"]
        self.text_chunks = []
        self.vector_store = None

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
        """Tải và xử lý tất cả các tài liệu trong thư mục."""
        all_chunks = []
        for filename in os.listdir(self.document_folder_path):
            file_path = os.path.join(self.document_folder_path, filename)
            if os.path.isfile(file_path) and filename.split(".")[-1].lower() in self.supported_formats:
                all_chunks.extend(self.load_and_process_document(file_path))
        self.text_chunks = all_chunks
        return all_chunks

    def create_vector_store(self):
        """Tạo vector store sử dụng HuggingFaceEmbeddings và Chroma."""
        if not self.text_chunks:
            raise ValueError("Chưa có text chunks. Vui lòng tải và xử lý tài liệu trước.")

        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = Chroma.from_documents(documents=self.text_chunks, embedding=hf_embeddings)
        return self.vector_store

