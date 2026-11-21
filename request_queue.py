"""FIFO inference queue for the single-process API server."""

import asyncio
import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


class QueueFull(Exception):
    pass


@dataclass
class QuestionJob:
    text: str
    image: bytes | None
    user_id: int | None
    conversation_id: str | None = None
    conversation: dict = field(default_factory=dict, repr=False)
    id: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    status: str = "queued"
    answer: str | None = None
    error: str | None = None
    forget_result: bool = False
    requester_key: str | None = None
    finished_at: float | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    submitted_at: float = field(default_factory=time.monotonic, repr=False)


class InferenceQueue:
    def __init__(self, processor: Callable[[QuestionJob], str], *,
                 max_pending: int = 20, result_ttl: float = 900,
                 initializer: Callable[[], None] | None = None,
                 finalizer: Callable[[QuestionJob], None] | None = None):
        self.processor = processor
        self.initializer = initializer
        self.finalizer = finalizer
        self.max_pending = max_pending
        self.result_ttl = result_ttl
        self._jobs: dict[str, QuestionJob] = {}
        self._pending: deque[str] = deque()
        self._active: str | None = None
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._closing = False
        self._preparing = initializer is not None

    async def start(self):
        self._worker = asyncio.create_task(self._run())

    async def close(self):
        self._closing = True
        while self._pending:
            job = self._jobs[self._pending.popleft()]
            job.status = "failed"
            job.error = "Сервер перезапускается. Отправьте запрос ещё раз."
            self._finish(job)
        self._wake.set()
        if self._worker:
            # Do not start another inference while a thread is still running.
            await self._worker

    def submit(self, text: str, image: bytes | None, user_id: int | None, *,
               conversation_id: str | None = None, conversation: dict | None = None,
               requester_key: str | None = None):
        self._prune()
        if self._closing or self._worker is None or self._worker.done():
            raise RuntimeError("Inference worker is not running")
        if len(self._pending) >= self.max_pending:
            raise QueueFull
        job = QuestionJob(text=text, image=image, user_id=user_id,
                          conversation_id=conversation_id, conversation=conversation or {}, requester_key=requester_key)
        self._jobs[job.id] = job
        self._pending.append(job.id)
        self._wake.set()
        return job

    def has_active_requester(self, requester_key):
        return any(job.requester_key == requester_key and job.status in {'queued', 'processing'}
                   for job in self._jobs.values())

    def authorize(self, job_id, user_id):
        job = self._jobs.get(job_id)
        if job is None or job.forget_result or job.user_id != user_id:
            raise KeyError(job_id)
        return job

    def forget_owner(self, user_id):
        ids = {job.conversation_id for job in self._jobs.values() if job.user_id == user_id}
        for conversation_id in ids:
            self.forget_conversation(conversation_id)

    def snapshot(self, job_id: str):
        self._prune()
        job = self._jobs[job_id]
        if job.forget_result:
            raise KeyError(job_id)
        ahead = 0
        if job.status == "queued":
            ahead = self._pending.index(job_id) + int(self._active is not None)
        return {
            "job_id": job.id,
            "conversation_id": job.conversation_id,
            "status": job.status,
            "ahead": ahead,
            "answer": job.answer,
            "error": job.error,
            "preparing": self._preparing,
        }

    def forget_conversation(self, conversation_id):
        """Drop queued/results; running model work finishes without retaining its result."""
        for job in list(self._jobs.values()):
            if job.conversation_id != conversation_id:
                continue
            job.forget_result = True
            if job.status == "queued":
                self._pending.remove(job.id)
                job.status = "failed"
                self._finish(job)
            elif job.status != "processing":
                self._jobs.pop(job.id, None)

    def _prune(self):
        now = time.monotonic()
        expired = [key for key, job in self._jobs.items()
                   if job.finished_at is not None
                   and now - job.finished_at >= self.result_ttl]
        for key in expired:
            del self._jobs[key]

    def _finish(self, job):
        # Photos are needed only during inference, not while storing the result.
        job.image = None
        job.text = ""
        job.conversation.clear()
        if self.finalizer is not None:
            self.finalizer(job)
        if job.forget_result:
            job.answer = None
            self._jobs.pop(job.id, None)
        job.finished_at = time.monotonic()
        job.done.set()

    async def _run(self):
        if self.initializer is not None:
            try:
                # Preparation shares this worker with inference: HTTP stays
                # responsive and no question races a second model load.
                await asyncio.to_thread(self.initializer)
            except Exception:
                logger.exception("Inference preparation failed; continuing with normal retrieval")
            finally:
                self._preparing = False
        while True:
            await self._wake.wait()
            if self._closing:
                return
            while self._pending:
                job = self._jobs[self._pending.popleft()]
                self._active = job.id
                job.status = "processing"
                try:
                    # There is exactly one worker. HTTP polling/disconnects do
                    # not cancel inference or allow another job to overtake it.
                    job.answer = await asyncio.to_thread(self.processor, job)
                    job.status = "done"
                except Exception:
                    logger.exception("Inference failed for job %s", job.id[:8])
                    job.status = "failed"
                    job.error = "Не удалось получить ответ от нейросети. Попробуйте ещё раз."
                finally:
                    self._finish(job)
                    self._active = None
                if self._closing:
                    return
            self._wake.clear()
