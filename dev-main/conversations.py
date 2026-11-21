"""Bounded, process-local conversation memory; IDs are secret bearer capabilities."""
import secrets
import threading
import time
from copy import deepcopy
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
    image_bytes: bytes | None = field(default=None, repr=False)
    image_saved_at: float = 0.0
    soil_case: dict = field(default_factory=dict)
    last_used: float = field(default_factory=time.monotonic)
    busy: bool = False
    active_job_id: str | None = None
    owner_id: int | None = None


class ConversationStore:
    def __init__(self, *, ttl=1800, max_conversations=100, max_turns=8, max_chars=24000,
                 clock=time.monotonic, image_ttl=1800, max_image_bytes=32 * 1024 * 1024):
        self.ttl = ttl
        self.image_ttl = image_ttl
        self.max_image_bytes = max_image_bytes
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
                if now - item.image_saved_at >= self.image_ttl:
                    item.image_bytes = None
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
                                    soil_case=memory.get('soil_case', {}),
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
            memory = {
                "history": [dict(turn) for turn in item.turns],
                "image_description": item.image_description,
            }
            if item.image_bytes:
                memory["remembered_image_bytes"] = item.image_bytes
            if item.soil_case:
                memory["soil_case"] = deepcopy(item.soil_case)
            return item.id, memory

    def bind_job(self, conversation_id, job_id):
        with self._lock:
            item = self._items.get(conversation_id)
            if item is not None:
                item.active_job_id = job_id

    def complete(self, conversation_id, question, answer, *, has_image=False, image_description=None,
                 soil_case=None, clear_photo=False, image_bytes=None):
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
            if has_image or clear_photo:
                item.image_bytes = image_bytes if has_image and isinstance(image_bytes, bytes) else None
                item.image_saved_at = self.clock()
                item.image_description = (image_description or "")[:4000]
            if isinstance(soil_case, dict):
                item.soil_case = deepcopy(soil_case)
            item.last_used = self.clock()
            # This is a RAM cache only. Never persist pixels to SQLite or expose
            # them through snapshots. Pending jobs have their own bounded input.
            total = sum(len(value.image_bytes or b'') for value in self._items.values())
            for value in sorted(self._items.values(), key=lambda row: row.last_used):
                if total <= self.max_image_bytes:
                    break
                total -= len(value.image_bytes or b'')
                value.image_bytes = None

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
