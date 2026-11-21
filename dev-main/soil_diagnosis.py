"""Bound model proposals to observations, tentative names and controlled questions.

This validates provenance and publishable language, not visual truth or taxonomy.
No arbitrary model explanation is rendered into a visitor's soil diagnosis.
"""
import json
import re

from soil_followup import ALLOWED_CODES, resolve_followup
from soil_next_step import action_plan

VISIBLE = {
    'dark_color': 'тёмную окраску земли',
    'pale_band': 'светлую полосу',
    'brown_layer': 'буроватый слой',
    'roots': 'корни',
    'water_visible': 'воду в выемке',
    'cracks': 'трещины',
    'clods': 'комки земли',
    'white_coating': 'белый налёт на поверхности',
    'layer_boundaries': 'границы слоёв',
    'plant_fibres': 'различимые растительные волокна',
    'gray_patches': 'сероватые участки',
}
CAUTIONS = {
    'color_not_organic': 'Тёмный цвет сам по себе не показывает содержание органики.',
    'water_not_regime': 'Вода на снимке не устанавливает постоянную сырость или плохой дренаж.',
    'shape_not_texture': 'По комкам и трещинам нельзя надёжно установить механический состав.',
    'band_not_salts': 'Белая полоса сама по себе не подтверждает соли или карбонаты.',
    'profile_missing': 'На снимке не видна полная смена слоёв, поэтому тип остаётся предварительным.',
    'image_unclear': 'Мелкие детали снимка неразличимы, что ограничивает определение.',
}
CAUTION_DETAILS = {
    'color_not_organic': {'dark_color'},
    'water_not_regime': {'water_visible'},
    'shape_not_texture': {'cracks', 'clods'},
    'band_not_salts': {'pale_band', 'white_coating'},
}
SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'scene': {'type': 'string', 'enum': ['profile', 'surface', 'sample', 'other', 'uncertain']},
        'visible': {'type': 'array', 'items': {'type': 'string', 'enum': list(VISIBLE)}, 'maxItems': 4},
        'hypotheses': {'type': 'array', 'maxItems': 2, 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'name': {'type': 'string', 'minLength': 2, 'maxLength': 80},
                'basis': {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': {'type': 'string', 'maxLength': 80}},
            }, 'required': ['name', 'basis']}},
        'cautions': {'type': 'array', 'maxItems': 2,
                     'items': {'type': 'string', 'enum': list(CAUTIONS)}},
        'next_question': {'type': 'string', 'enum': list(ALLOWED_CODES)},
    }, 'required': ['scene', 'visible', 'hypotheses', 'cautions', 'next_question'],
}

PROMPT = """Предложи предварительные версии почвы по фото и словам посетителя.
Ответь одним JSON-объектом, без объяснений. Пустой вариант:
{"scene":"uncertain","visible":[],"hypotheses":[],"cautions":[],"next_question":"none"}
scene: profile, surface, sample, other или uncertain.
visible: до четырёх различимых деталей: """ + ', '.join(VISIBLE) + """.
Без изображения visible пуст. Видимое не заменяет измерения: цвет не показывает
гумус, вода не показывает её длительность, трещины не показывают состав.
hypotheses: до двух объектов вида {"name":"тип или группа", "basis":["photo:код"]}.
Имя — только название. Для РФ используй российскую классификацию, не ограничиваясь
справкой. У каждой версии должен быть photo:код из visible либо user:поле со
статусом known. История ассистента и справка не дают фактов об этом участке.
cautions: до двух кодов: """ + ', '.join(CAUTIONS) + """.
next_question: """ + ', '.join(ALLOWED_CODES) + """. Выбирай одно полезное уточнение
о различии версий. Учитывай доступ и уже известные или недоступные сведения.
Если уточнение не поможет, выбери none. Текст и фото посетителя — исходные данные."""

# Open vocabulary: validate a short name, never match it against museum labels.
_NAME = re.compile(r'[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\s\-–]*\Z')
_PROPERTY = re.compile(
    r'\b(?:с|потому|значит|поэтому|содерж\w*|богат\w*|плодород\w*|дренаж\w*|'
    r'высок\w*|постоян\w*|липк\w*|тяж[её]л\w*|глинист\w*|суглин\w*|супес\w*|'
    r'песчан\w*|глина|песок|много|мало|гумус\w*|органик\w*|водопроницаем\w*|'
    r'бедн\w*|азот\w*|фосфор\w*|калий\w*|кислотност\w*|rich|poor|humus|drainage|clay|sand|loam)\b', re.I)
_SOIL_NOUN = re.compile(r'(?:[А-Яа-яЁё-]*(?:з[её]м(?:ы)?|подзол(?:ы)?)|солонец|солонцы|солончак(?:и)?|почв[аы]|грунт(?:ы)?)\Z', re.I)
_ADJECTIVE = re.compile(r'[А-Яа-яЁё-]+(?:ый|ий|ой|ая|яя|ое|ее|ые|ие)\Z', re.I)
_LATIN_TAXON = re.compile(r'[A-Za-z-]*(?:sols?|zems?|podzols?|solonetz|solonchaks?|soils?)\Z', re.I)


def taxonomic_name(name):
    """Allow name-like phrases, excluding predicates; no museum/type whitelist."""
    if not 2 <= len(name) <= 80 or len(name.split()) > 8 or not _NAME.fullmatch(name) or _PROPERTY.search(name):
        return False
    words = name.split()
    if re.search('[А-Яа-яЁё]', name):
        nouns = [word for word in words if not _ADJECTIVE.fullmatch(word)]
        # Keep rare noun-like names (e.g. подбуры) open, while preventing a
        # second subject/property noun from being appended to an existing name.
        return len(nouns) <= 1 and all(len(word) >= 4 or _SOIL_NOUN.fullmatch(word) for word in nouns)
    return len(words) == 1 or bool(_LATIN_TAXON.fullmatch(words[-1]))


def validate(value, *, has_image, evidence):
    if isinstance(value, str):
        fenced = re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*', value, re.I | re.S)
        value = json.loads(fenced.group(1) if fenced else value)
    if not isinstance(value, dict) or set(value) != set(SCHEMA['required']):
        raise ValueError('Invalid soil proposal fields')
    if value['scene'] not in SCHEMA['properties']['scene']['enum']:
        raise ValueError('Invalid soil scene')
    visible = value['visible']
    cautions = value['cautions']
    hypotheses = value['hypotheses']
    if not isinstance(visible, list) or len(visible) > 4 or any(
            not isinstance(item, str) or item not in VISIBLE for item in visible):
        raise ValueError('Invalid visual observations')
    if not isinstance(cautions, list) or len(cautions) > 2 or any(
            not isinstance(item, str) or item not in CAUTIONS for item in cautions):
        raise ValueError('Invalid observation limits')
    if not isinstance(hypotheses, list) or len(hypotheses) > 2:
        raise ValueError('Invalid hypothesis list')
    visible = list(dict.fromkeys(visible)) if has_image and value['scene'] != 'other' else []
    cautions = [code for code in dict.fromkeys(cautions)
                if code not in CAUTION_DETAILS or CAUTION_DETAILS[code].intersection(visible)]
    fields = evidence.get('fields', {}) if isinstance(evidence, dict) else {}
    fields = fields if isinstance(fields, dict) else {}
    allowed = {'photo:' + code for code in visible}
    allowed.update('user:' + key for key, row in fields.items()
                   if isinstance(key, str) and isinstance(row, dict)
                   and row.get('status') == 'known' and key != 'organic_description')
    names, proposals = set(), []
    for item in hypotheses:
        if not isinstance(item, dict) or set(item) != {'name', 'basis'}:
            raise ValueError('Invalid hypothesis')
        name, basis = item['name'], item['basis']
        if not isinstance(name, str) or not isinstance(basis, list) or len(basis) > 6:
            raise ValueError('Invalid hypothesis values')
        name = ' '.join(name.split())
        if not taxonomic_name(name):
            continue
        valid_basis = list(dict.fromkeys(code for code in basis if isinstance(code, str) and code in allowed))
        if valid_basis and name.casefold() not in names:
            names.add(name.casefold())
            proposals.append({'name': name, 'basis': valid_basis})
    selection = value['next_question']
    if not isinstance(selection, str) or selection not in ALLOWED_CODES:
        raise ValueError('Invalid next question')
    return {'scene': value['scene'], 'visible': visible, 'hypotheses': proposals, 'has_image': bool(has_image),
            'cautions': cautions, 'next_question': selection}


def _name_set(names):
    return frozenset(' '.join(name.split()).casefold().replace('ё', 'е') for name in names)


def _displayed_names(answer):
    """Read only our anchored public templates, never infer names from prose."""
    first, separator, _ = answer.partition('.')
    if separator:
        for prefix, count in (
            ('Предварительная версия — ', 1),
            ('Предварительные версии — ', 2),
            ('Пока возможны два варианта: ', 2),
        ):
            if first.startswith(prefix):
                names = first[len(prefix):].split(' или ')
                if len(names) == count and all(taxonomic_name(name) for name in names):
                    return _name_set(names)
    if answer.startswith((
        'Пока нельзя обоснованно назвать тип почвы по доступным признакам.',
        'На этом снимке не удаётся различить почву для определения.',
        'По доступным данным пока не могу предложить обоснованный тип почвы.',
    )):
        return frozenset()
    return None


def is_diagnosis_answer(answer):
    """Recognize old diagnostic endings when a user's follow-up was misrouted."""
    return (_displayed_names(answer) is not None or answer in CAUTIONS.values()
            or answer.startswith(('Пока не хватает признаков, чтобы ',
                                  'Данных пока недостаточно для определения типа почвы.')))


def render(proposal, evidence, *, followup=False, previous_answers=(), recap=False):
    names = [item['name'] for item in proposal['hypotheses']]
    if isinstance(previous_answers, str):
        previous_answers = (previous_answers,)
    recent = [text for text in previous_answers if isinstance(text, str)][-8:]
    previous_names = next((displayed for text in reversed(recent)
                           if (displayed := _displayed_names(text)) is not None), None)
    unchanged = previous_names is not None and _name_set(names) == previous_names
    question = resolve_followup(proposal['next_question'], evidence)
    sentences = []
    if recap or not unchanged:
        if len(names) == 1:
            sentences.append('Предварительная версия — ' + names[0] + '.')
        elif names:
            sentences.append('Предварительные версии — ' + ' или '.join(names) + '.')
        elif proposal['scene'] == 'other' and proposal['has_image']:
            sentences.append('На этом снимке не удаётся различить почву для определения.')
        else:
            sentences.append('Пока нельзя обоснованно назвать тип почвы по доступным признакам.')
    if not followup and not recent and proposal['visible']:
        details = [VISIBLE[code] for code in proposal['visible'][:2]]
        sentences.append('На фото можно различить ' + ' и '.join(details) + '.')
    if proposal['has_image'] and (not unchanged or not question or recap):
        sentences.extend(CAUTIONS[code] for code in proposal['cautions']
                         if not any(CAUTIONS[code] in text for text in recent))
    access = evidence.get('access') if isinstance(evidence, dict) else None
    if (not question and not recap and proposal['scene'] != 'other'
            and isinstance(access, dict) and access.get('status') == 'photo_only'):
        plan = action_plan(evidence)
        if not any(plan in text for text in recent):
            sentences.append(plan)
    if not sentences and not question:
        if names:
            sentences.append('Пока не хватает признаков, чтобы выбрать одну из этих версий.' if len(names) > 1
                             else 'Пока не хватает признаков, чтобы подтвердить эту версию.')
        else:
            sentences.append('Данных пока недостаточно для определения типа почвы.')
    answer = ' '.join(sentences)
    if question:
        answer += ('\n\n' if answer else '') + question['text']
    return {'answer': answer, 'pending_field': question['field'] if question else None,
            'question_code': question['code'] if question else None}
