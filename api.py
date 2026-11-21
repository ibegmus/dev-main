import os
import asyncio
from contextlib import suppress
import time
import re
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Response, Request, Query
from files import extract_text, ALLOWED_EXT, MAX_SIZE, describe_photo, decode_photo
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from http_security import SecurityHeaders
from password_security import password_problem
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uvicorn

import db
import usage
from conversations import (ConversationStore, ConversationExpired, ConversationBusy, ConversationCapacity)
from request_queue import InferenceQueue, QueueFull, QuestionJob
from retrieval import preload_reranker
from performance import request_profile, timed_stage
from account_security import (SESSION_COOKIE, SESSION_AGE, TERMS_VERSION, WindowLimiter,
                              check_browser_mutation, secure_cookie, guest_identity)

# Ключи лучше задавать через переменные окружения, а не хранить в коде
ADMIN_KEY = os.getenv("ADMIN_KEY")

@asynccontextmanager
async def lifespan(app):
    db.init_db()
    usage.recover()
    conversations = ConversationStore()
    app.state.conversations = conversations
    app.state.limiter = WindowLimiter()
    def finish_job(job):
        usage.finish(getattr(job, "quota_id", None), job.status == "done")
        conversations.release(job.conversation_id)
    queue = InferenceQueue(process_question, initializer=preload_reranker,
                           finalizer=finish_job)
    app.state.question_queue = queue
    await queue.start()
    async def sweep_memory():
        while True:
            await asyncio.sleep(30)
            conversations.prune()
            queue._prune()
    sweeper = asyncio.create_task(sweep_memory())
    try:
        yield
    finally:
        await queue.close()
        sweeper.cancel()
        with suppress(asyncio.CancelledError):
            await sweeper
        conversations.clear()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

app.add_middleware(SecurityHeaders)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Pydantic includes raw input values by default, including rejected passwords.
    errors = [{key: item[key] for key in ('type', 'loc', 'msg') if key in item} for item in exc.errors()]
    return JSONResponse(status_code=422, content={'detail': errors})


splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

# ---------- Модели запросов ----------

class QuestionRequest(BaseModel):
    text: str = Field(default="", max_length=4000)
    conversation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{43}$")
    image: str | None = None   # JPEG в base64, без префикса data:


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=128)
    terms_accepted: bool = False


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class DeleteAccountRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=12, max_length=128)


class KnowledgeRequest(BaseModel):
    title: str
    content: str


# ---------- Проверки доступа ----------

def check_admin_key(x_admin_key):
    if not ADMIN_KEY or not isinstance(x_admin_key, str) or not secrets.compare_digest(x_admin_key, ADMIN_KEY):
        raise HTTPException(status_code=403, detail="Нет доступа")


# ---------- Пользователи ----------

def credentials(request):
    cookie = request.cookies.get(SESSION_COOKIE)
    header = request.headers.get('x-token')
    if cookie and header and cookie != header:
        raise HTTPException(401, 'Неоднозначные данные входа. Войдите заново.')
    return header or cookie


def current_user(request, required=False):
    token = credentials(request)
    user = db.get_user_by_token(token)
    if token and user is None:
        raise HTTPException(401, 'Срок входа истёк. Войдите заново.')
    if required and user is None:
        raise HTTPException(401, 'Нужно войти в аккаунт.')
    return user


def auth_response(user, token, request, response):
    response.headers['Cache-Control'] = 'no-store'
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_AGE, httponly=True,
                        secure=secure_cookie(request), samesite='lax')
    result = {'user': dict(user)}
    # Browser code uses only the HttpOnly cookie; keep legacy tokens for CLI clients.
    if not request.headers.get('origin') and not request.headers.get('sec-fetch-site'):
        result['token'] = token
    return result


def auth_limit(request, email, action):
    app.state.limiter.check((action, email.strip().casefold()), limit=10, seconds=900,
                           message='Слишком много попыток. Попробуйте через 15 минут.')
    peer = request.client.host if request.client else 'unknown'
    app.state.limiter.check((action, 'peer', peer), limit=60, seconds=60,
                           message='Слишком много запросов входа. Попробуйте через минуту.')


@app.post('/register', status_code=201)
def register(req: RegisterRequest, request: Request, response: Response):
    check_browser_mutation(request)
    name, email = req.name.strip(), req.email.strip().lower()
    if not name or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
        raise HTTPException(400, 'Укажите имя и корректный email.')
    if req.terms_accepted is not True:
        raise HTTPException(400, 'Подтвердите принятие условий сервиса.')
    auth_limit(request, email, 'register')
    problem = password_problem(req.password, email)
    if problem:
        raise HTTPException(400, problem)
    user_id = db.create_user(name, email, req.password, terms_version=TERMS_VERSION)
    if user_id is None:
        raise HTTPException(409, 'Этот email уже зарегистрирован.')
    token = db.login(email, req.password)
    return auth_response(db.get_user_by_token(token), token, request, response)


@app.post('/login')
def login(req: LoginRequest, request: Request, response: Response):
    check_browser_mutation(request)
    auth_limit(request, req.email, 'login')
    token = db.login(req.email, req.password)
    if token is None:
        raise HTTPException(401, 'Неверный email или пароль.')
    return auth_response(db.get_user_by_token(token), token, request, response)


@app.get('/me')
def me(request: Request, response: Response):
    response.headers['Cache-Control'] = 'no-store'
    return {'user': dict(current_user(request, required=True))}


@app.post('/logout', status_code=204)
def logout(request: Request, response: Response):
    check_browser_mutation(request)
    token = credentials(request)
    if token:
        db.revoke_session(token)
    response.delete_cookie(SESSION_COOKIE, httponly=True, secure=secure_cookie(request), samesite='lax')
    response.headers['Cache-Control'] = 'no-store'


@app.delete('/account', status_code=204)
async def delete_account(req: DeleteAccountRequest, request: Request, response: Response):
    check_browser_mutation(request)
    user = current_user(request, required=True)
    auth_limit(request, user['email'], 'delete-account')
    if not db.verify_password(user['id'], req.password):
        raise HTTPException(403, 'Неверный пароль.')
    db.delete_user(user['id'])
    app.state.conversations.delete_owner(user['id'])
    app.state.question_queue.forget_owner(user['id'])
    response.delete_cookie(SESSION_COOKIE, httponly=True, secure=secure_cookie(request), samesite='lax')
    response.headers['Cache-Control'] = 'no-store'


@app.post('/account/password')
def change_password(req: ChangePasswordRequest, request: Request, response: Response):
    check_browser_mutation(request)
    user = current_user(request, required=True)
    auth_limit(request, user['email'], 'change-password')
    problem = password_problem(req.new_password, user['email'])
    if problem:
        raise HTTPException(400, problem)
    if req.current_password == req.new_password:
        raise HTTPException(400, 'Новый пароль должен отличаться от текущего.')
    token = db.change_password(user['id'], req.current_password, req.new_password)
    if token is None:
        raise HTTPException(403, 'Не удалось изменить пароль. Проверьте текущий пароль и повторите попытку.')
    return auth_response(user, token, request, response)


@app.get('/history')
def history(request: Request, response: Response):
    user = current_user(request, required=True)
    response.headers['Cache-Control'] = 'no-store'
    return {'history': db.get_history(user['id'])}


# ---------- Вопрос нейросети ----------

def run_agent(text: str, image_bytes, conversation=None):
    # Model loading belongs to the worker, so the API remains responsive while
    # the first request loads the local models.
    with timed_stage("agent_init"):
        from agent import app as agent_app

    state = {"question": text}
    if conversation:
        if conversation.get("history"):
            state["history"] = conversation["history"]
        if conversation.get("image_description"):
            state["image_description"] = conversation["image_description"]
    if image_bytes:
        state["image_description"] = describe_photo(image_bytes)
        if state.get("history"):
            state["new_image"] = True
    result = agent_app.invoke(state)
    if image_bytes:
        result["image_description"] = state["image_description"]
    return result


def process_question(job: QuestionJob):
    with request_profile(job.id, job.submitted_at):
        start_time = time.monotonic()
        if job.conversation.get("history") or job.conversation.get("image_description"):
            result = run_agent(job.text, job.image, job.conversation)
        else:
            result = run_agent(job.text, job.image)
        answer = result["answer"]
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("The model returned an empty answer")
        saved_question = job.text + (" [с фото]" if job.image else "")
        if job.user_id is not None:
            saved = db.save_conversation_turn(
                job.conversation_id, job.user_id, job.text, answer, result.get("intent"),
                time.monotonic() - start_time, bool(job.image), result.get("image_description"))
            if not saved:
                return answer  # Deleted account/chat must never be recreated by a late answer.
        if job.conversation_id:
            app.state.conversations.complete(
                job.conversation_id, job.text, answer, has_image=bool(job.image),
                image_description=result.get("image_description"))
        return answer


def submit_question(req: QuestionRequest, request: Request, response: Response):
    check_browser_mutation(request)
    user = current_user(request)
    user_id = user["id"] if user else None

    requester_key = ('user:' + str(user_id)) if user else 'guest:' + guest_identity(request, response)
    if app.state.question_queue.has_active_requester(requester_key):
        raise HTTPException(429, 'Ваш предыдущий запрос ещё обрабатывается. Дождитесь ответа.',
                            headers={'Retry-After': '3'})
    if req.conversation_id is None:
        peer = request.client.host if request.client else 'unknown'
        app.state.limiter.check(('new-chat', requester_key), limit=5, seconds=600,
                               message='Можно начать до 5 новых чатов за 10 минут. Продолжите существующий чат или подождите.')
        if user is None:
            app.state.limiter.check(('guest-new-peer', peer), limit=60, seconds=60,
                                   message='Слишком много новых гостевых чатов. Попробуйте через минуту.')
    image_bytes = None
    if req.image:
        try:
            image_bytes = decode_photo(req.image)
        except OverflowError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    text = req.text.strip()
    if not text and not image_bytes:
        raise HTTPException(status_code=400, detail="Напишите вопрос или прикрепите фото.")
    if not text:
        text = "Что изображено на этом фото? Какая это может быть почва?"

    try:
        conversation_id, memory = app.state.conversations.reserve(req.conversation_id, owner_id=user_id)
    except ConversationExpired as exc:
        raise HTTPException(status_code=404 if user else 410, detail="Беседа недоступна или её память истекла. Начните новый чат.") from exc
    except ConversationBusy as exc:
        raise HTTPException(status_code=409, detail="В этой беседе уже готовится ответ. Дождитесь его или начните новый чат.") from exc
    except ConversationCapacity as exc:
        raise HTTPException(status_code=429, detail="Сейчас открыто слишком много бесед. Попробуйте позже.", headers={"Retry-After": "30"}) from exc
    created = req.conversation_id is None and user_id is not None
    if created and not db.create_conversation(conversation_id, user_id, text[:80]):
        app.state.conversations.delete(conversation_id)
        raise HTTPException(429, 'Сохранено 50 чатов. Удалите ненужный чат перед созданием нового.')
    quota_id = None
    try:
        try:
            quota_id = usage.reserve(requester_key, user_id)
        except ValueError as error:
            raise HTTPException(429, str(error)) from error
        job = app.state.question_queue.submit(text, image_bytes, user_id,
                                              conversation_id=conversation_id, conversation=memory,
                                              requester_key=requester_key)
        job.quota_id = quota_id
        app.state.conversations.bind_job(conversation_id, job.id)
        return job
    except Exception as exc:
        usage.finish(quota_id, False)
        app.state.conversations.release(conversation_id)
        if req.conversation_id is None:
            app.state.conversations.delete(conversation_id)
            if created:
                db.delete_conversation(conversation_id, user_id)
        if not isinstance(exc, QueueFull):
            raise

        raise HTTPException(status_code=429, detail="Очередь заполнена. Попробуйте чуть позже.",
                            headers={"Retry-After": "10"}) from exc


@app.post('/ask/jobs', status_code=202)
async def enqueue_question(req: QuestionRequest, request: Request, response: Response):
    response.headers['Cache-Control'] = 'no-store'
    job = submit_question(req, request, response)
    return app.state.question_queue.snapshot(job.id)


@app.get('/ask/jobs/{job_id}')
async def question_status(job_id: str, request: Request, response: Response):
    response.headers['Cache-Control'] = 'no-store'
    user = current_user(request)
    try:
        app.state.question_queue.authorize(job_id, user['id'] if user else None)
        return app.state.question_queue.snapshot(job_id)
    except KeyError as exc:
        raise HTTPException(404, 'Запрос не найден. Возможно, сервер перезапустился.') from exc



class FeedbackRequest(BaseModel):
    conversation_id: str = Field(pattern=r"^[A-Za-z0-9_-]{43}$")
    stars: int = Field(ge=1, le=5, strict=True)
    comment: str = Field(min_length=10, max_length=2000)

@app.get('/account/usage')
async def account_usage(request: Request, response: Response):
    user = current_user(request)
    owner = 'user:' + str(user['id']) if user else 'guest:' + guest_identity(request, response)
    return usage.status(owner, user['id'] if user else None)

@app.post('/feedback', status_code=201)
async def submit_feedback(req: FeedbackRequest, request: Request):
    check_browser_mutation(request)
    user = current_user(request, required=True)
    try:
        usage.submit_feedback(user['id'], req.conversation_id, req.stars, req.comment)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return usage.status('user:' + str(user['id']), user['id'])

@app.get('/admin/feedback')
async def admin_feedback(request: Request, before: int = Query(default=2147483647, ge=1)):
    user = current_user(request, required=True)
    if user['role'] != 'admin':
        raise HTTPException(403, 'Доступ только администратору.')
    return {'feedback': usage.feedback_list(before)}

@app.get('/conversations')
async def list_conversations(request: Request, response: Response):
    user = current_user(request, required=True)
    response.headers['Cache-Control'] = 'no-store'
    return {'conversations': db.list_conversations(user['id']), 'has_more': False}


@app.get('/conversations/{conversation_id}')
async def conversation_status(conversation_id: str, request: Request, response: Response,
                              before: int | None = Query(default=None, ge=1)):
    response.headers['Cache-Control'] = 'no-store'
    user = current_user(request)
    owner_id = user['id'] if user else None
    if user:
        page = db.get_conversation_turns(conversation_id, owner_id, before=before)
        if page is None:
            raise HTTPException(404, 'Беседа не найдена.')
        try:
            snapshot = app.state.conversations.snapshot(conversation_id, owner_id)
        except ConversationCapacity:
            snapshot = {'active_job_id': None}
        return {'conversation_id': conversation_id, **page, 'persisted': True,
                'active_job_id': snapshot.get('active_job_id')}
    # A saved chat always requires its account, even when its identifier is known.
    try:
        return app.state.conversations.snapshot(conversation_id)
    except ConversationExpired as exc:
        raise HTTPException(410, 'Память этой беседы истекла или была очищена. Начните новый чат.',
                            headers={'Cache-Control': 'no-store'}) from exc


@app.delete('/conversations/{conversation_id}', status_code=204)
async def delete_conversation(conversation_id: str, request: Request, response: Response):
    check_browser_mutation(request)
    user = current_user(request)
    if user:
        if not db.delete_conversation(conversation_id, user['id']):
            raise HTTPException(404, 'Беседа не найдена.')
    else:
        try:
            app.state.conversations.snapshot(conversation_id)
        except ConversationExpired as exc:
            raise HTTPException(410, 'Беседа недоступна.') from exc
    app.state.conversations.delete(conversation_id)
    app.state.question_queue.forget_conversation(conversation_id)
    response.headers['Cache-Control'] = 'no-store'


@app.post('/ask')
async def ask(req: QuestionRequest, request: Request, response: Response):
    response.headers['Cache-Control'] = 'no-store'
    job = submit_question(req, request, response)
    await job.done.wait()
    if job.status == 'failed' or job.forget_result:
        raise HTTPException(503, job.error or 'Запрос был отменён.')
    return {'answer': job.answer}


# ---------- Знания для нейросети (только админ) ----------

@app.post("/knowledge")
def add_knowledge(req: KnowledgeRequest, x_admin_key: str = Header(None)):
    check_admin_key(x_admin_key)
    from agent import vectorstore, build_bm25

    # 1. Сохраняем исходный текст в SQLite
    knowledge_id = db.add_knowledge(req.title, req.content)

    # 2. Режем на куски и добавляем в Chroma, чтобы нейросеть могла это находить
    chunks = splitter.split_text(req.content)
    vectorstore.add_texts(
        texts=chunks,
        metadatas=[{"knowledge_id": knowledge_id, "title": req.title}] * len(chunks),
    )
    build_bm25()
    return {"id": knowledge_id, "chunks": len(chunks)}


@app.get("/knowledge")
def list_knowledge(x_admin_key: str = Header(None)):
    check_admin_key(x_admin_key)
    return {"items": db.list_knowledge()}

@app.post("/upload")
def upload_file(file: UploadFile = File(...), x_admin_key: str = Header(None)):
    check_admin_key(x_admin_key)
    from agent import vectorstore, build_bm25

    # 1. Проверка типа и размера
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Поддерживаются PDF, DOCX, TXT, MD, PNG, JPG")

    data = file.file.read(MAX_SIZE + 1)
    if len(data) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Файл больше 10 МБ")

    # 2. Извлечение текста
    try:
        text = extract_text(file.filename, data)
    except Exception as e:
        print(f"❌ ОШИБКА ЧТЕНИЯ ФАЙЛА: {e}")
        raise HTTPException(status_code=400, detail="Не удалось прочитать файл")

    if len(text) < 50:
        raise HTTPException(status_code=422, detail="В файле нет текста (возможно, это скан)")

    # 3. Сохраняем и индексируем, как в /knowledge
    title = os.path.basename(file.filename)
    knowledge_id = db.add_knowledge(title, text)
    chunks = splitter.split_text(text)
    vectorstore.add_texts(
        texts=chunks,
        metadatas=[{"knowledge_id": knowledge_id, "title": title}] * len(chunks),
    )
    build_bm25()
    return {"id": knowledge_id, "title": title, "chunks": len(chunks)}

# Добавление монолитов без правки кода
class MonolithRequest(BaseModel):
    name: str
    region: str | None = None
    year: int | None = None
    soil_type: str | None = None
    humus: float | None = None
    ph: float | None = None
    description: str | None = None


@app.post("/monoliths")
def add_monolith(req: MonolithRequest, x_admin_key: str = Header(None)):
    check_admin_key(x_admin_key)
    monolith_id = db.add_monolith(**req.model_dump())
    return {"id": monolith_id}

# Статика (сайт) подключается последней, иначе перехватит остальные маршруты
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("Запускаем сервер. Подготовка поиска будет выполняться в фоне.", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
