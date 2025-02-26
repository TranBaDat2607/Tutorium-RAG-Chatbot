from vanna.remote import VannaDefault
import re
import os
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
VANNA_DB_API_KEY = os.getenv("DATABASE_VANNA_API")
VANNA_DB_MODEL_NAME = os.getenv("VANNA_DB_MODEL_NAME")
DB_PORT = os.getenv("DATABASE_PORT")

class DatabaseAgent:
    def __init__(self, db_user: str, db_password: str, db_host: str, db_database: str, vanna_model_name: str, db_port: str, llm_apiKey: str) -> None:
        self.llm = VannaDefault(model=vanna_model_name, api_key=llm_apiKey)

    def clean_sql_query(self, query: str) -> str:
        """Loại bỏ mô tả và chỉ giữ lại truy vấn SQL hợp lệ."""
        query = re.sub(r"```sql|```", "", query, flags=re.IGNORECASE).strip()
        return query

    def llm_gen_sqlquery(self, userInput: str) -> str:
        """
        Dùng Vanna AI để phân tích và tạo ra câu lệnh MySQL.

        Args:
            userInput: input được gửi từ user.

        Returns:
            str: câu lệnh truy vấn MySQL.
        """
        mySQL_query = self.llm.generate_sql(userInput)
        return self.clean_sql_query(mySQL_query)

db_agent = DatabaseAgent(
        db_user=DB_USER,
        db_password=DB_PASSWORD,
        db_host=DB_HOST,
        db_database=DB_NAME,
        vanna_model_name=VANNA_DB_MODEL_NAME,
        db_port=DB_PORT,
        llm_apiKey=DB_API_KEY
    )

print(db_agent.llm_gen_sqlquery("có bao nhiêu người dùng"))