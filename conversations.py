"""Bounded, process-local conversation memory; IDs are secret bearer capabilities."""
import secrets
import threading
import time
from dataclasses import dataclass, field


class ConversationExpired(Exception):
    pass


class ConversationBusy(Exception):
    pass


class ConversationCapacity(Exception):
    pass


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    turns: list = field(default_factory=list)
    image_description: str = ""
    last_used: float = field(default_factory=time.monotonic)
    busy: bool = False
    active_job_id: str | None = None
    owner_id: int | None = None


class ConversationStore:
    def __init__(self, *, ttl=1800, max_conversations=100, max_turns=8, max_chars=24000,
                 clock=time.monotonic):
        self.ttl = ttl
        self.max_conversations = max_conversations
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.clock = clock
        self._items = {}
        self._lock = threading.RLock()

    def prune(self):
        with self._lock:
            now = self.clock()
            for key, item in list(self._items.items()):
                if not item.busy and now - item.last_used >= self.ttl:
                    del self._items[key]

    def _get(self, conversation_id, owner_id=None):
        item = self._items.get(conversation_id)
        if item is not None:
            if item.owner_id != owner_id:
                raise ConversationExpired
            return item
        if owner_id is not None:
            import db
            memory = db.get_conversation_memory(conversation_id, owner_id)
            if memory is not None:
                if len(self._items) >= self.max_conversations:
                    # Saved chats remain in SQLite; evict idle RAM copies only.
                    idle = next((key for key, value in self._items.items() if not value.busy and value.owner_id is not None), None)
                    if idle is None:
                        raise ConversationCapacity
                    del self._items[idle]
                item = Conversation(id=conversation_id, owner_id=owner_id,
                                    turns=memory['history'], image_description=memory['image_description'],
                                    last_used=self.clock())
                self._items[item.id] = item
                return item
        raise ConversationExpired

    def reserve(self, conversation_id=None, owner_id=None):
        with self._lock:
            self.prune()
            if conversation_id:
                item = self._get(conversation_id, owner_id)
            else:
                if len(self._items) >= self.max_conversations:
                    raise ConversationCapacity
                item = Conversation(last_used=self.clock(), owner_id=owner_id)
                self._items[item.id] = item
            if item.busy:
                raise ConversationBusy
            item.busy = True
            item.last_used = self.clock()
            return item.id, {
                "history": [dict(turn) for turn in item.turns],
                "image_description": item.image_description,
            }

    def bind_job(self, conversation_id, job_id):
        with self._lock:
            item = self._items.get(conversation_id)
            if item is not None:
                item.active_job_id = job_id

    def complete(self, conversation_id, question, answer, *, has_image=False, image_description=None):
        with self._lock:
            item = self._items.get(conversation_id)
            # Deleting a conversation while inference runs must never recreate it.
            if item is None:
                return
            item.turns.append({"question": question[:4000], "answer": answer[:6000],
                               "has_image": has_image})
            while len(item.turns) > self.max_turns or sum(
                    len(t["question"]) + len(t["answer"]) for t in item.turns) > self.max_chars:
                item.turns.pop(0)
            if has_image:
                item.image_description = (image_description or "")[:4000]
            item.last_used = self.clock()

    def release(self, conversation_id):
        with self._lock:
            item = self._items.get(conversation_id)
            if item is not None:
                item.busy = False
                item.active_job_id = None
                item.last_used = self.clock()

    def snapshot(self, conversation_id, owner_id=None):
        with self._lock:
            self.prune()
            item = self._get(conversation_id, owner_id)
            # Reading/polling doesn't extend retention indefinitely.
            return {"conversation_id": item.id,
                    "turns": [dict(turn) for turn in item.turns],
                    "active_job_id": item.active_job_id,
                    "persisted": item.owner_id is not None}

    def delete(self, conversation_id):
        with self._lock:
            self._items.pop(conversation_id, None)

    def delete_owner(self, owner_id):
        with self._lock:
            for key in [key for key, item in self._items.items() if item.owner_id == owner_id]:
                del self._items[key]

    def clear(self):
        with self._lock:
            self._items.clear()
