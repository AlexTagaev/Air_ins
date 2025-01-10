# импорт библиотек
from fastapi import FastAPI     # библиотека FastAPI
from pydantic import BaseModel  # модуль для объявления структуры данных
from api.chunks import Chunk    # модуль для работы с OpenAI
from api.chunks import load_requests_count

# создаем объект приложения FastAPI
app = FastAPI()

# создадим объект для работы с OpenAI
chunk = Chunk()

# класс с типами данных для метода api/get_answer
class ModelAnswer(BaseModel):
    text: str    

# функция, которая будет обрабатывать запрос по пути "/"
# полный путь запроса http://127.0.0.1:8000/
@app.get("/")
def root(): 
    return {"message": "Hello FastAPI"}

# функция, которая обрабатывает запрос по пути "/about"
@app.get("/about")
def about():
    return {"message": "Страница API нейро-консультанта"}

# функция, которая обрабатывает запрос по пути "/stat"
@app.get("/api/stat")
def stat():
    total_requests = load_requests_count()
    return {'Total requests': total_requests}

# функция обработки post запроса + декоратор 
@app.post('/api/get_answer')
def get_answer(question: ModelAnswer):
    answer = chunk.get_answer(query = question.text)
    return {'Answer': answer[0], 'Total requests': answer[1]}    