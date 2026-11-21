"""Deterministic interview policy; field reports are not confirmed soil diagnoses."""
import re
from soil_observations import enrich_fields, normalize_fields, quote_is_statement

QUESTIONS = {
    'location': 'В каком регионе и рядом с каким населённым пунктом находится участок? Точный адрес не нужен.',
    'landscape': 'Что сейчас находится вокруг этого места: лес, луг, поле, сад или что-то другое?',
    'water': 'После дождя вода здесь быстро уходит или долго стоит на поверхности?',
    'origin': 'Известно ли, завозили сюда грунт, торф или компост?',
    'white_layer': 'Светлое видно как налёт на поверхности или как отдельный слой внутри разреза?',
    'organic_layer': 'В тёмном слое различимы растительные остатки или он больше похож на обычную минеральную землю?',
    'texture': 'Если уже пробовали увлажнять щепотку земли, она рассыпается или держит шарик и короткую ленту? Если не пробовали, так и скажите.',
    'lower_layer': 'Что видно под светлым слоем: рыжевато-бурый рыхлый материал или заметно более плотная глинистая земля?',
    'clay_change': 'На уже доступном срезе нижний слой заметно глинистее верхнего, а на гранях комочков видны гладкие глинистые плёнки? Если не можете проверить, так и скажите.',
    'soil_structure': 'Как устроена земля на уже видимом срезе: мелкие зернистые комочки, вертикальные столбики, крупные глыбки или сплошная масса? Укажите, в каком слое вы это видите.',
    'redox': 'Уже видимые нижние слои сизо-голубые или серо-зелёные с ржавыми пятнами?',
    'organic_depth': 'Если это уже измеряли, какая мощность именно торфяного слоя с растительными остатками и в каких единицах? Для ответа не нужно копать глубокую яму.',
    'salinity': 'Что указано в готовом анализе водорастворимых солей или электропроводности — результат, метод, единицы и глубина образца? Если анализа нет, скажите об этом.',
    'sodicity': 'Что указано в готовом анализе обменного натрия и других обменных катионов, с единицами и глубиной отбора? Если анализа нет, скажите об этом.',
    'carbonate': 'Есть ли уже описание специалиста или анализ карбонатов в слоях ниже гумусового горизонта? Налёт и реакция с уксусом сами по себе это описание не заменяют.',
    'profile': 'Есть ли фото уже доступного среза с линейкой, где видна смена слоёв? Копать глубокую яму не нужно.',
}
LABELS = {
    'location':'Место','landscape':'Растительность и использование','water':'Увлажнение',
    'origin':'Привозной грунт','white_layer':'Светлый слой','organic_layer':'Органический слой',
    'texture':'Механический состав','profile':'Профиль','tests':'Измерения и анализы',
    'lower_layer':'Материал под светлым слоем','clay_change':'Изменение глинистости',
    'soil_structure':'Структура','redox':'Сизые и ржавые участки','organic_depth':'Мощность торфяной толщи',
    'salinity':'Анализ засоления','sodicity':'Обменный натрий','carbonate':'Карбонатный горизонт',
    'location_detail':'Уточнение населённого пункта','surface_water':'Вода на снимке',
    'vegetation':'Видимая растительность','stones':'Камни',
}
UNKNOWN = re.compile(r'не знаю|неизвест|не (?:проверял|измерял|могу|помню|делал)|нет (?:данных|анализ(?:а|ов))|анализ(?:а|ов) нет|без анализ(?:а|ов)', re.I)
FINISH = re.compile(r'подвед\w*.{0,20}итог|сдела\w*.{0,20}вывод|больше (?:сведений|данных|информации).{0,20}нет|хватит.{0,15}вопрос|достаточно вопросов', re.I)
MAX_QUESTIONS = 3

EXTRACTION_PROMPT = '''
Для определения участка добавь observations: список до 5 объектов
{"field":"location","status":"known","quote":"Свердловская область"}.
Поля: location, location_detail, landscape, vegetation, water, surface_water, stones,
origin, white_layer, organic_layer, texture, profile, tests,
lower_layer, clay_change, soil_structure, redox, organic_depth, salinity, sodicity, carbonate.
Сохраняй конкретные наблюдения: где расположен светлый слой, что ниже него, структуру,
мощность органического слоя с единицами и результат уже выполненного анализа.
Не выводи эти свойства из одного цвета, города или своей догадки о типе.
Сохраняй известную страну/регион в location, неизвестный населённый пункт — в
location_detail. Вода на одном снимке — surface_water, её обычная длительность — water.
Комочки и камни не являются механическим составом: это soil_structure и stones.
Используй ТОЛЬКО сведения последнего сообщения посетителя, не историю и не фото.
quote — дословная короткая цитата из сообщения. Не дополняй её своими выводами.
known означает «посетитель сообщил», а не «проверено». Если он пишет «не знаю»,
«не проверял», «анализов нет» — status=unknown для соответствующего поля.
Если сведений нет, поле пропусти. На вопрос посетителя не записывай ответ от его имени.
'''

def normalize(text):
    return ' '.join(text.casefold().split())

def validated_updates(question, updates):
    result = {}
    if not isinstance(updates, list):
        return result
    for row in updates[:5]:
        if not isinstance(row, dict):continue
        key, status, quote = row.get('field'), row.get('status'), row.get('quote')
        if not isinstance(key,str) or key not in LABELS or not isinstance(status,str) or status not in {'known','unknown'} or not isinstance(quote,str):continue
        quote = ' '.join(quote.split())
        if not 2 <= len(quote) <= 300 or '?' in quote or normalize(quote) not in normalize(question):continue
        if key not in {'location','location_detail'} and not quote_is_statement(question,quote):continue
        if status == 'unknown' and not UNKNOWN.search(quote):continue
        if status == 'known' and UNKNOWN.search(quote):continue
        if status=='unknown' and result.get(key,{}).get('status')=='known':continue
        result[key] = {'status':status,'quote':quote}
    return result

def direct_answer(field, text):
    patterns = {
        'location': r'област|край|республик|город|рядом с',
        'landscape': r'лес|луг|поле|сад|степь|болот|тайг|пойм|трава',
        'water': r'стоит|уходит|впит|сух|мокр|луж|заста|недел|после дождя|всегда|обычно',
        'origin': r'^(?:да|нет)$|привоз|завоз|насып',
        'white_layer': r'нал[её]т|слой|поверхност|внутри|глубин',
        'organic_layer': r'растительн|остатк|минеральн|волокн|корн',
        'texture': r'шарик|лент|рассып|песок|песчан|липк|крош',
        'profile': r'фото|срез|линейк|профил',
        'lower_layer': r'ниже|бур|рыж|глинист|плотн|песок',
        'clay_change': r'глинист|пленк|кутан|больше глин',
        'soil_structure': r'зернист|комковат|глыб|столбчат|столбик|призм',
        'redox': r'сиз|голубоват|серо-зелен|ржав',
        'organic_depth': r'\d+(?:[.,]\d+)?\s*(?:см|мм|метр|м\b)',
        'salinity': r'анализ|лаборатор|водорастворим|электропровод',
        'sodicity': r'анализ|лаборатор|обменн.*натри',
        'carbonate': r'карбонат|анализ|специалист',
    }
    pattern=patterns.get(field)
    if not pattern:return False
    return bool(re.search(pattern,text,flags=0 if field=='location' else re.I))

def update_fields(previous, question, updates):
    previous = previous if isinstance(previous,dict) else {}
    fields = {}
    for key, row in previous.get('fields', {}).items() if isinstance(previous.get('fields'),dict) else []:
        if key in LABELS and isinstance(row,dict) and isinstance(row.get('status'),str) and row['status'] in {'known','unknown'} and isinstance(row.get('quote'),str):
            fields[key]={'status':row['status'],'quote':row['quote'][:300]}
    fields=normalize_fields(fields)
    incoming=enrich_fields(question,validated_updates(question,updates),fields)
    pending = previous.get('pending_field')
    if not isinstance(pending,str):pending=None
    # A direct short answer inherits only the single question the program asked.
    short = ' '.join(question.split()).strip(' .!')
    if pending in QUESTIONS and re.fullmatch(r'(?:я )?(?:не знаю|не помню|не проверял|не могу проверить|неизвестно)',short,re.I):
        fields[pending]={'status':'unknown','quote':short}
        return fields
    fields.update(incoming)
    if pending in QUESTIONS and not incoming and short:
        if re.fullmatch(r'(?:я )?(?:не знаю|не помню|не проверял|не могу проверить|неизвестно)',short,re.I):
            fields[pending]={'status':'unknown','quote':short}
        elif len(short)<=120 and '?' not in short and not FINISH.search(short) and not UNKNOWN.search(short) and direct_answer(pending, short):
            # Keep the raw answer, never interpret it as a measured value/type.
            fields[pending]={'status':'known','quote':short}
    return normalize_fields(fields)

def choose_question(case, photo=''):
    asked=[k for k in case.get('asked_fields',[]) if k in QUESTIONS]
    if len(asked)>=MAX_QUESTIONS:return None
    from soil_knowledge import evaluate
    from soil_questions import choose
    return choose(case,evaluate(case),photo)

def reply(case, question, photo=''):
    fields=case.get('fields',{})
    lines=['Соберём признаки участка. Тип почвы по одному цвету или фотографии пока не установлен.']
    if fields:
        lines.append('Из ваших сообщений:')
        for key,row in fields.items():
            lines.append(f"• {LABELS[key]}: {row['quote']}" + (' (неизвестно; повторно не спрашиваю)' if row['status']=='unknown' else ' (сообщено вами)'))
    if photo:
        lines.append('Наблюдение по фото, которое может ошибаться: '+photo[:800])
    from soil_photo import confirmation_summary
    confirmed = confirmation_summary(case)
    if confirmed:
        lines.append('Проверка видимых признаков с вами:\n'+confirmed)
    from soil_knowledge import evaluate, render
    assessment=evaluate(case)
    next_field = None if FINISH.search(question) else choose_question(case,photo)
    if assessment['evidence']['imported'] or assessment['evidence']['conflicts'] or not next_field:
        lines.append(render(assessment))
    elif assessment['candidates']:
        lines.append('Есть сочетание признаков для справочных гипотез; сначала уточним различающий признак. Тип почвы пока не установлен.')
    updated=dict(case)
    updated.pop('pending_photo',None)
    if next_field:
        from soil_photo import confirmation_question
        from soil_questions import photo_question
        visual = photo_question(case, assessment, next_field)
        updated['asked_fields']=[*case.get('asked_fields',[]),next_field][-len(QUESTIONS):]
        updated['pending_field']=next_field
        updated['active']=True
        if visual:
            updated['pending_photo']={'photo_id':visual['photo_id'],'feature':visual['feature']}
        lines.append('Следующее уточнение: '+(confirmation_question(visual) if visual else QUESTIONS[next_field]))
    else:
        updated.pop('pending_field',None)
        updated['active']=False
        lines.append('Предварительный итог: диагностическая группа не установлена; для окончательного вывода недостаточно проверенных диагностических данных. '
                     'Это не означает, что почва плохая или непригодна для растений. '
                     'Песчаность, влажность и цвет — признаки, а не готовый тип почвы; pH, гумус и NPK по фото не измеряются.')
        from soil_questions import final_step
        lines.append(final_step(case,assessment))
    return '\n\n'.join(lines), updated
