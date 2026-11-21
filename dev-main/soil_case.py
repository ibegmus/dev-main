"""Small, source-labelled memory for a soil identification interview.

Only the visitor's words and descriptions of submitted photos are recorded here.
Earlier assistant answers are deliberately excluded from the evidence.
"""

from soil_interview import QUESTIONS, update_fields, validated_updates
from soil_photo import record as record_photo, awaiting_confirmation, answer_status
from soil_observations import record_features
from soil_evidence import record as record_evidence

MAX_USER_NOTES = 8
MAX_PHOTOS = 4


def durable_evidence(previous, question, updates):
    """Migrate visitor notes once; never seed facts from legacy model fields."""
    evidence = previous.get("evidence")
    if not (isinstance(evidence, dict) and evidence.get("kind") == "user_soil_evidence"
            and evidence.get("version") == 1):
        evidence = {}
        notes = previous.get("user_notes", [])
        for note in notes[-MAX_USER_NOTES:] if isinstance(notes, list) else []:
            if isinstance(note, str):
                evidence = record_evidence(evidence, note)
    # Only the controlled production reply writes pending fields inside evidence.
    # A legacy top-level pending field never identifies a free-form model question.
    memory = {key: evidence[key] for key in ("kind", "version", "access", "fields")
              if key in evidence}
    source = {"evidence": memory}
    pending = evidence.get("pending_field")
    if isinstance(pending, str) and pending in set(QUESTIONS) | {"access"}:
        source["pending_field"] = pending
    result = record_evidence(source, question, validated_updates(question, updates))
    asked = evidence.get("asked_fields", [])
    result["asked_fields"] = list(dict.fromkeys(
        key for key in asked if isinstance(key, str) and key in set(QUESTIONS) | {"access"}
    ))[-16:] if isinstance(asked, list) else []
    return result


def record_case(previous, question, photo_description="", *, new_photo=False, reset=False, updates=None, photo_observation=None):
    previous = previous if isinstance(previous, dict) and not reset else {}
    old_notes = previous.get("user_notes", [])
    old_photos = previous.get("photos", [])
    user_notes = [" ".join(value.split())[:1000] for value in old_notes
                  if isinstance(value, str)] if isinstance(old_notes, list) else []
    photos = [" ".join(value.split())[:1200] for value in old_photos
              if isinstance(value, str)] if isinstance(old_photos, list) else []
    if question.strip():
        user_notes.append(" ".join(question.split())[:1000])
    if new_photo and photo_description.strip():
        photos.append(" ".join(photo_description.split())[:1200])
    # A bare yes/no belongs only to the exact visual question, never to a field
    # invented by the classifier (e.g. yes must not mean imported ground).
    if not new_photo and awaiting_confirmation(previous) and answer_status(question):
        updates = None
    result = {"version": 4, "user_notes": user_notes[-MAX_USER_NOTES:],
              "photos": photos[-MAX_PHOTOS:], "fields": update_fields(previous, question, updates),
              "evidence": durable_evidence(previous, question, updates)}
    features = record_features(previous.get('reported_features', []), question)
    if features:result['reported_features'] = features
    photo_memory = record_photo(previous, question, new_photo=new_photo, observation=photo_observation)
    if photo_memory['photo_observations']:
        result.update(photo_memory)
    result['asked_fields'] = [key for key in previous.get('asked_fields', []) if isinstance(key,str) and key in QUESTIONS][:8] if isinstance(previous.get('asked_fields'),list) else []
    if isinstance(previous.get('pending_field'),str):result['pending_field']=previous['pending_field']
    return result


def case_context(case):
    if not isinstance(case, dict):
        return ""
    lines = [f"Сообщение посетителя {i}: {note}" for i, note in enumerate(case.get("user_notes", []), 1)]
    lines += [f"Описание фото {i} (наблюдение модели, может ошибаться): {photo}"
              for i, photo in enumerate(case.get("photos", []), 1)]
    return "\n".join(lines)
