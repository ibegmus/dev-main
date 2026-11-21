import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from model_runtime import keep_alive_seconds

# --- НАСТРОЙКИ ---
DOC_PATH = os.getenv("RAG_DOC_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "Video Project 9.txt"))
DB_PATH = os.getenv("RAG_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db_new"))

embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3"),
    keep_alive=keep_alive_seconds(), client_kwargs={"timeout": 30.0},
)

vectorstore = None 

def index_document():
    global vectorstore
    
    if not os.path.exists(DOC_PATH):
        print(f"Ошибка: Файл {DOC_PATH} не найден!")
        return

    print("Загрузка документа...")
    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Сначала бьем по заголовкам (Markdown)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(text)

    # 2. Затем добиваем, если куски остались слишком большими
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(md_header_splits)
    
    # Очищаем старую базу, чтобы не дублировать
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)

    print(f"Создание эмбеддингов для {len(chunks)} кусков...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"✅ Готово! База пересоздана.")


def ask_question(question: str):
    # CLI and website use the same retrieval, prompts and web fallback.
    from agent import app
    return app.invoke({"question": question})["answer"]

if __name__ == "__main__":
    # Раскомментируй, чтобы пересоздать базу (ОБЯЗАТЕЛЬНО СДЕЛАЙ ЭТО 1 РАЗ)
    #index_document() 
    
    # Закомментируй и проверь
    question = "Какие почвы преобладают в Кытлымском районе?"
    print(f"Вопрос: {question}")
    print("Ответ:", ask_question(question))
