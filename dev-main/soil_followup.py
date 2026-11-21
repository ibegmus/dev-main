"""Render one controlled follow-up without trusting free-form model instructions."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Followup:
    fields: tuple[str, ...]
    text: str
    needs_access: bool = False
    minutes: int = 0
    cost_rub: int = 0


FOLLOWUPS = {
    'access': Followup(('access',), 'У вас сейчас есть доступ к участку или образец земли под рукой?'),
    'location': Followup(('location',), 'Если знаете, в каком регионе находится участок?'),
    'landscape': Followup(('landscape',),
                         'Если знаете, что окружает место: лес, луг, поле или сад?'),
    'imported': Followup(('origin', 'imported'),
                        'Если знаете, сюда завозили грунт или это местная земля?'),
    'water_regime': Followup(('water', 'water_regime'),
                            'Если знаете по прежним наблюдениям, вода после дождя быстро уходит или стоит несколько дней?'),
    'material': Followup(('organic_layer', 'material'),
                        'За 1 минуту без инструментов рассмотрите щепотку земли: преобладают узнаваемые растительные волокна и остатки или минеральные песчинки и комочки?',
                        needs_access=True, minutes=1),
    'texture': Followup(('texture',),
                       'За 2 минуты без инструментов возьмите щепотку земли, добавьте пару капель воды и попробуйте пальцами скатать шарик и короткую ленту. Земля рассыпается, держит только шарик или получается и лента?',
                       needs_access=True, minutes=2),
    'profile': Followup(('profile',),
                       'Если рядом уже есть открытый срез земли, без копки опишите за 1 минуту порядок видимых слоёв сверху вниз. Какие цвета и слои вы различаете?',
                       needs_access=True, minutes=1),
}
ALLOWED_CODES = (*FOLLOWUPS, 'none')
INFO_ORDER = ('location', 'landscape', 'imported')
NOT_TRIED = re.compile(r'\bне\s+(?:проверял[аи]?|пробовал[аи]?|трогал[аи]?)\b', re.I)
CANNOT_TRY = re.compile(r'\bне\s+могу\b', re.I)


def resolve_followup(selection, evidence):
    """Return one safe question with its actual code/field, or ``None``.

    Only ``user_soil_evidence`` v1 fields establish known/unknown facts. Program
    asked/pending markers may sit in that record or its enclosing soil case.
    An unavailable model choice is omitted; unknown access falls back to one
    unanswered information question before suggesting any physical observation.
    An explicit on-site user who has not tried a simple material/texture check
    may receive that experiment once; asked/pending markers still prevent repeats.
    """
    if not isinstance(selection, str) or selection not in FOLLOWUPS:
        return None
    outer = evidence if isinstance(evidence, dict) else {}
    nested = outer.get('evidence')
    memory = nested if isinstance(nested, dict) else outer
    if memory.get('kind') != 'user_soil_evidence' or memory.get('version') != 1:
        memory = {}
    fields = memory.get('fields')
    fields = fields if isinstance(fields, dict) else {}
    access = memory.get('access')
    access = access.get('status') if isinstance(access, dict) else 'unknown'
    asked = set()
    for source in (outer, memory):
        entries = source.get('asked_fields')
        if isinstance(entries, (list, tuple)):
            asked.update(entry for entry in entries if isinstance(entry, str))
        pending = source.get('pending_field')
        if isinstance(pending, str):
            asked.add(pending)

    def available(code):
        followup = FOLLOWUPS[code]
        if code == 'access' and access in ('photo_only', 'on_site'):
            return False
        if code in asked or any(field in asked for field in followup.fields):
            return False
        for field in followup.fields:
            entry = fields.get(field)
            if not isinstance(entry, dict):
                continue
            if entry.get('status') == 'known':
                return False
            if entry.get('status') == 'unknown':
                quote = entry.get('quote')
                if not (code in ('material', 'texture') and access == 'on_site'
                        and isinstance(quote, str) and NOT_TRIED.search(quote)
                        and not CANNOT_TRY.search(quote)):
                    return False
        return True

    if not available(selection):
        return None
    followup = FOLLOWUPS[selection]
    if followup.needs_access:
        if access == 'photo_only':
            return None
        if access != 'on_site':
            selection = next((code for code in (*INFO_ORDER, 'access') if available(code)), None)
            if selection is None:
                return None
            followup = FOLLOWUPS[selection]
    return {'text': followup.text, 'code': selection, 'field': followup.fields[0]}


def choose_followup(selection, evidence):
    """Compatibility wrapper returning only the resolved question text."""
    resolved = resolve_followup(selection, evidence)
    return resolved['text'] if resolved else ''
