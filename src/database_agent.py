from typing import List, Dict, Any, Union
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sqlalchemy import inspect, text


class DatabaseAgent:
    def __init__(self, db_user: str, db_password: str, db_host: str, db_database: str, llm_apiKey: str) -> None:
        """
        Khởi tạo đối tượng DatabaseAgent với các thông số kết nối cơ sở dữ liệu.

        Args:
            db_user (str): Tên người dùng của cơ sở dữ liệu.
            db_password (str): Mật khẩu của cơ sở dữ liệu.
            db_host (str): Host của cơ sở dữ liệu.
            db_database (str): Tên cơ sở dữ liệu.
            llm_apiKey (str): API key cho LLM.
        """
        self.database: Engine = create_engine(
            f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_database}",
            pool_pre_ping=True,
            echo=False
        )
        self.database_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=llm_apiKey
        )
        self.system_prompt = (
            "You are an expert SQL assistant. Your task is to generate a valid MySQL query based solely on the user's request. "
            "Do not include any additional explanations, commentary, or markdown formatting. "
            "Your output must be a single, executable SQL query without any surrounding text. "
            "If the query is complete and ready for execution, it must start exactly with the marker 'FINAL QUERY:'. "
            "For example, a correct output should be: \n"
            "FINAL QUERY: SELECT * FROM users WHERE id = 1;\n\n"
            "Below is the database schema information for your reference:\n"
            "Tables: {tables}\n"
            "Foreign key relationships: {fks}\n"
            "Table columns (as a dictionary with table names as keys and lists of column names, column datatypes and column value example): {columns}\n"
            "You can only FINAL SELECT this attribute: {allow_cols}"
        ).format(
            tables=self.takeDatabaseTables(),
            fks=self.takeDatabaseConnections(),
            columns=self.getTableColumns(),
            allow_cols=self.allow_cols

        )
        self.allow_cols = {
        "Users.username",
        "Users.displayname",
        "Users.imgurl",

        "Roles.rolename",
        "Roles.description",

        "Permissions.slug",
        "Permissions.name",
        "Permissions.description",

        "Tutors.description",
        "Tutors.descriptionvideolink",

        "Learners.learninggoal",

        "Accomplishments.description",
        "Accomplishments.verifylink",

        "Categories.categoryname",
        "Categories.description",

        "Contracts.target",
        "Contracts.timestart",
        "Contracts.timeend",
        "Contracts.payment",
        "Contracts.status",

        "WorkingTimes.starttime",
        "WorkingTimes.endtime",
        "WorkingTimes.note",

        "Posts.title",
        "Posts.content",
        "Posts.posttime",

        "Comments.content",
        "Comments.comment_time",
        }


    def takeDatabaseTables(self) -> List[str]:
        """
        Lấy danh sách tất cả các bảng trong cơ sở dữ liệu.

        Returns:
            List[str]: Danh sách tên các bảng.
        """
        inspector = inspect(self.database)
        return inspector.get_table_names()

    def takeDatabaseConnections(self) -> List[dict]:
        """
        Lấy danh sách tất cả các ràng buộc khóa ngoại giữa các bảng.

        Returns:
            List[dict]: Danh sách các kết nối giữa bảng với thông tin khóa ngoại.
        """
        inspector = inspect(self.database)
        connections = []

        for table in inspector.get_table_names():
            foreign_keys = inspector.get_foreign_keys(table)
            for fk in foreign_keys:
                connections.append({
                    "table": table,
                    "column": fk["constrained_columns"],
                    "referenced_table": fk["referred_table"],
                    "referenced_column": fk["referred_columns"]
                })

        return connections

    def getTableColumns(self) -> Dict[str, List]:
        """
        Lấy thông tin danh sách các thuộc tính (tên cột) của từng bảng trong cơ sở dữ liệu,
        kiểu dữ liệu của các cột và một số ví dụ giá trị trong bảng.

        Ví dụ:
            Bảng Users gồm có UserID, Username,...
            Hàm này sẽ trả về:
            {
                "Users": [
                    {"UserID", "Username"},                  # Tập hợp tên cột
                    {"INTEGER", "VARCHAR"},                    # Tập hợp kiểu dữ liệu (dạng chuỗi)
                    {"UserID": "1, 2, 3", "Username": "Nguyễn Văn A, Nguyễn Văn B, Trần Quốc C"}  # Ví dụ của giá trị từng cột
                ]
            }

        Returns:
            Dict[str, List]: Một dictionary với key là tên bảng và value là danh sách gồm:
                             - Tập hợp tên cột (set[str])
                             - Tập hợp kiểu dữ liệu (set[str])
                             - Dictionary ví dụ giá trị của các cột (Dict[str, str])
        """
        inspector = inspect(self.database)
        table_info = {}

        for table in inspector.get_table_names():
            columns = inspector.get_columns(table)

            # Tập hợp tên cột và kiểu dữ liệu (được chuyển thành chuỗi)
            col_names = {col["name"] for col in columns}
            col_types = {str(col["type"]) for col in columns}

            sample_values = {}
            with self.database.connect() as conn:
                # Sử dụng sqlalchemy.text để bọc truy vấn và mappings() để nhận kết quả dạng dictionary
                query = text(f"SELECT * FROM {table} LIMIT 3")
                rows = conn.execute(query).mappings().all()

            if rows:
                for col in columns:
                    col_name = col["name"]
                    samples = [str(row[col_name]) for row in rows if row[col_name] is not None]
                    sample_values[col_name] = ", ".join(samples) if samples else ""
            else:
                for col in columns:
                    sample_values[col["name"]] = ""

            table_info[table] = [col_names, col_types, sample_values]
        print(table_info)
        return table_info

    def execute_query(self, query: str) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        """
        Thực thi câu truy vấn SQL và trả về kết quả dưới dạng danh sách các dictionary.

        Args:
            query (str): Câu truy vấn SQL cần thực thi.

        Returns:
            Union[List[Dict[str, Any]], Dict[str, str]]:
            - Danh sách các dictionary chứa kết quả của truy vấn,
            - Hoặc một dictionary chứa thông báo lỗi nếu truy vấn không thành công.
        """
        try:
            with self.database.connect() as conn:
                result = conn.execute(text(query)).fetchall()  # Lấy tất cả kết quả của truy vấn
                # Chuyển mỗi dòng kết quả thành một dictionary sử dụng thuộc tính _mapping
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error("Query execution failed", exc_info=True)
            return {"error": f"Query execution failed: {str(e)}"}



    def clean_sql_query(self, query: str) -> str:
        """Loại bỏ mô tả và chỉ giữ lại truy vấn SQL hợp lệ."""
        # Loại bỏ markdown nếu có
        query = re.sub(r"sql|", "", query, flags=re.IGNORECASE).strip()
        # Nếu có FINAL QUERY: thì chỉ lấy phần sau đó
        if "FINAL QUERY:" in query:
            query = query.split("FINAL QUERY:")[-1].strip()
        else:
            # Nếu không có marker, có thể chỉ lấy dòng đầu tiên
            query = query.splitlines()[0].strip()
        print(query)
        return query

    def llm_gen_sqlquery(self, userInput: str) -> str:
        """
        Dùng model llm để phân tích và tạo ra câu lệnh MySQL

        Args:
            userInput: input được gửi từ user

        Returns:
            str: câu lệnh truy vấn MySQL
        """
        helper_prompt = (
            "Based on the complete database schema provided, please analyze the following user input and generate an optimized, complex, and valid MySQL query. "
            "Ensure that every attribute used in the query is fully qualified in the format 'table_name.attribute_name' (e.g., 'Users.username', 'Orders.OrderID'). "
            "Your query should handle complex conditions, multiple joins, aggregations, subqueries, and any other advanced SQL features as needed, while strictly adhering to MySQL syntax. "
            "Do not include any additional commentary or extraneous text in your output. "
            "User input: "
        )
        mySQL_query = self.database_llm.invoke(self.system_prompt + helper_prompt + userInput).content
        return mySQL_query

    def make_SQLresponse_natural(self, sqlResponse: Union[List[Dict[str, Any]], Dict[str, str]], userInput: str) -> AIMessage:
        helper_prompt = (
                "Bạn là một trợ lý AI chuyên nghiệp, nhiệm vụ của bạn là dựa vào câu hỏi của người dùng và chuyển đổi kết quả truy vấn SQL thô thành một phản hồi tự nhiên, "
                "dễ hiểu và thân thiện cho người dùng bằng tiếng Việt. "
                "Nếu truy vấn trả về dữ liệu, hãy trả lời theo dạng: 'Hệ thống hiện có 5 người dùng.' và kết thúc bằng '<<END>>'. "
                "Một số dạng trả về đặc biệt ví dụ như: [{'Có_dạy_ai_không': 'Có'}] sẽ phải dựa vào câu hỏi của người dùng để phản hồi 'và kết thúc bằng '<<END>> "
                "Nếu không có dữ liệu, hãy trả lời dựa theo câu hỏi của người dùng một cách hợp lý và kết thúc bằng '<<END>>'.\n"
                f"Đây là câu hỏi của người dùng: {userInput}"
                f"Kết quả truy vấn SQL: {sqlResponse}"
        )
        print(sqlResponse)
        natural_response = self.database_llm.invoke(helper_prompt).content
        # Trả về đối tượng AIMessage với tham số content đã được gán
        return AIMessage(content=natural_response)
