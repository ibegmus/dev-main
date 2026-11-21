"""Small practical continuations for an already active soil diagnosis.

The caller owns topic detection and whether a plan was already displayed.
These helpers do not classify soil, infer access, or prescribe treatment.
"""
import re


_FILLERS = frozenset({
    'ну', 'так', 'ладно', 'ок', 'окей', 'хорошо', 'да', 'а',
    'блин', 'блять', 'блядь', 'сука', 'нахуй', 'ебать',
})
_PRACTICAL = tuple(re.compile(pattern) for pattern in (
    r'(?:и )?(?:что|че|чо)(?: (?:дальше|теперь))?',
    r'(?:и )?(?:что|че|чо)(?: мне(?: то)?)?(?: теперь)? делать(?: (?:дальше|теперь))?',
    r'мне(?: то)? (?:что|че|чо)(?: теперь)? делать(?: (?:дальше|теперь))?',
    r'(?:и )?(?:что|че|чо)(?: мне)? с этим делать',
    r'мне(?: то)? (?:что|че|чо) с этим делать',
    r'(?:и )?(?:что )?дальше',
    r'(?:какой )?следующий шаг',
))

_OBSERVATION = (
    'за 2 минуты сделайте один снимок уже открытого среза земли сбоку при дневном свете: '
    'пусть в кадр попадут поверхность и слои ниже. '
    'По нему можно сравнить, выражены ли отдельные слои или материал выглядит однородным. '
    'Если открытого среза нет, специально копать и что-либо покупать не надо.'
)
_CHAT_PLAN = (
    'Не меняйте кислотность и не вносите удобрения только на основании этой версии почвы. '
    'Если хотите разобраться с практической задачей прямо в чате, напишите, '
    'что собираетесь выращивать или какую проблему уже заметили.'
)


def practical_followup(text):
    """Match short requests for the next step, without swallowing a new topic."""
    if not isinstance(text, str) or not text.strip() or len(text) > 160:
        return False
    normalized = re.sub(r'[^\w\s]', ' ', text.casefold().replace('ё', 'е'))
    normalized = ' '.join(word for word in normalized.split() if word not in _FILLERS)
    return any(pattern.fullmatch(normalized) for pattern in _PRACTICAL)


def _access(evidence):
    if not isinstance(evidence, dict):
        return 'unknown'
    nested = evidence.get('evidence')
    memory = nested if isinstance(nested, dict) else evidence
    if memory.get('kind') != 'user_soil_evidence' or memory.get('version') != 1:
        return 'unknown'
    access = memory.get('access')
    status = access.get('status') if isinstance(access, dict) else None
    return status if status in ('on_site', 'photo_only') else 'unknown'


def action_plan(evidence, *, followup=False):
    """Return one bounded observation plan, or its practical chat continuation.

    An existing exposed profile can add evidence about the sequence of layers;
    a hand-texture result alone does not separate e.g. Podzols and Psammozems.
    No particular pair or laboratory property is inferred from absent context.
    """
    if followup:
        return _CHAT_PLAN
    access = _access(evidence)
    if access == 'photo_only':
        return 'Сейчас ничего проверять не нужно. Когда будете на участке, ' + _OBSERVATION
    if access == 'on_site':
        return 'Если рядом уже есть открытый срез земли, ' + _OBSERVATION
    return 'Когда будет возможность посмотреть участок, ' + _OBSERVATION
