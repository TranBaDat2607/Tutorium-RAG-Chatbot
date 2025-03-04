from typing import List, Dict, Any, Union
import logging
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine


class DatabaseManager:
    def __init__(self, db_user: str, db_password: str, db_host: str, db_database: str) -> None:
        self.database: Engine = create_engine(
            f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_database}",
            pool_pre_ping=True,
            echo=False
        )

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
        return table_info