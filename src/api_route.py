from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_init import process_input_message

app = FastAPI()

# Model cho request nhận tin nhắn từ client
class UserMessage(BaseModel):
    content: str

@app.post("/process")
async def process_message(message: UserMessage):
    # Gọi hàm xử lý tin nhắn qua đồ thị và trả về phần content của tin nhắn cuối cùng
    response_content = process_input_message(message.content)
    return {"response": response_content}

@app.get("/")
async def read_root():
    return {"message": "API server is running"}

if __name__ == "__main__":
    import uvicorn
    # Chạy server trên localhost với port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
