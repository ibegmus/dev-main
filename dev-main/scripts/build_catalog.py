"""Render reviewed soil data as static pages; no database or model is touched.

Run from any directory: python scripts/build_catalog.py
Use --check to verify committed pages are up to date, without writing files.
"""
import argparse
from html import escape
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'static'
START = '<!-- SOIL_CATALOG_START -->'
END = '<!-- SOIL_CATALOG_END -->'


def h(value):
    return escape(str(value), quote=True)


def url(soil):
    return f'/soils/{soil["id"]}/'


def photo(soil, role, *, lazy=True):
    asset = soil['images'][role]
    caption = ('Монолит' if role == 'monolith' else 'Музейная табличка') + ': ' + soil['title']
    if role == 'monolith' and soil['image_status'] == 'needs_review':
        caption = 'Фотография из исходной пары; принадлежность почве требует сверки'
    return (f'<img src="{h(asset["src"])}" alt="{h(caption)}" '
            f'width="{asset["width"]}" height="{asset["height"]}" '
            f'loading="{"lazy" if lazy else "eager"}" decoding="async">')


def facts(items):
    return '<dl class="soil-facts">' + ''.join(
        f'<div><dt>{h(item["label"])}</dt><dd>{h(item["value"])}</dd></div>'
        for item in items) + '</dl>'


def card(soil):
    terms = ' '.join([soil['id'], soil['title'], soil['family'], soil['summary']]
                     + [item['value'] for item in soil['facts']])
    badge = '<span class="review-badge">Фото требует сверки</span>' if soil['image_status'] == 'needs_review' else ''
    return f'''<a class="soil-card" href="{url(soil)}" data-family="{h(soil['family'])}" data-search="{h(terms)}">
  <div class="soil-image">{photo(soil, 'monolith')}<span class="specimen-number">{soil['id']}</span></div>
  <div class="soil-content"><span class="soil-family">{h(soil['family'])}</span>
    <h3>{h(soil['title'])}</h3><p>{h(soil['summary'])}</p>{badge}
    <div class="soil-highlights">{''.join(f'<span>{h(s["label"])} <strong>{h(s["value"])}</strong></span>' for s in soil['highlights'])}</div>
    <span class="soil-open">Подробнее <span aria-hidden="true">→</span></span>
  </div>
</a>'''


def catalog(soils):
    options = ''.join(f'<option value="{h(f)}">{h(f)}</option>' for f in sorted({s['family'] for s in soils}))
    return f'''{START}
<section class="catalog-section" id="catalog" aria-labelledby="catalog-heading">
  <div class="catalog-heading"><div><p class="catalog-eyebrow">Коллекция музея</p>
    <h2 id="catalog-heading">Почвы в деталях</h2><p>{len(soils)} образцов · фотографии монолитов и музейных табличек</p></div>
    <a class="catalog-download" href="/data/catalog.txt" download>Скачать описания ↓</a></div>
  <details class="catalog-filter-panel" hidden open>
    <summary>Найти образец <span>Название, место или группа</span></summary>
  <div class="catalog-controls" hidden>
    <label class="catalog-search">Поиск по коллекции<input id="catalog-search" type="search" placeholder="Название, место или год…" autocomplete="off"></label>
    <label>Группа почв<select id="catalog-family"><option value="">Все группы</option>{options}</select></label>
    <button class="catalog-reset" type="button" id="catalog-reset">Сбросить</button>
  </div>
  </details>
  <p class="catalog-mobile-hint">Листайте коллекцию <span aria-hidden="true">← →</span></p>
  <p class="catalog-count" id="catalog-count" role="status" aria-live="polite">В коллекции {len(soils)} образцов</p>
  <div class="catalog-grid" tabindex="0" role="region" aria-label="Карточки почвенных образцов">{''.join(card(s) for s in soils)}</div>
  <p id="catalog-empty" class="catalog-empty" hidden>Образцы не найдены. Попробуйте другое название или сбросьте фильтры.</p>
  <div class="catalog-more"><button id="catalog-more" type="button" hidden>Показать ещё</button></div>
</section>
{END}'''


def detail(soil, previous, following):
    notes = ''
    if soil['notes']:
        notes = '<aside class="source-notes"><h2>Примечания к источнику</h2>' + ''.join(
            f'<p>{h(n)}</p>' for n in soil['notes']) + '</aside>'
    horizons = ''
    if soil['horizons']:
        rows = ''.join(f'<tr><th scope="row">{h(r["name"])}</th><td>{h(r["depth_cm"])}</td></tr>' for r in soil['horizons'])
        horizons = f'''<section class="detail-section"><h2>Горизонты профиля</h2>
<p>Обозначения и интервалы глубины перенесены с экспозиционного листа.</p>
<table><caption>Глубина от поверхности, см</caption><thead><tr><th scope="col">Горизонт</th><th scope="col">Глубина, см</th></tr></thead><tbody>{rows}</tbody></table></section>'''
    scales = ''
    if soil.get('chart_scales'):
        scales = f'''<details class="source-details"><summary>Обозначения и шкалы графиков</summary>
<p>Ниже перечислены напечатанные отметки шкал, а не измеренные значения почвы. Кривые доступны на фотографии листа.</p>{facts(soil['chart_scales'])}</details>'''
    image_caption = 'Монолит целиком'
    if soil['image_status'] == 'needs_review':
        image_caption = 'Фотография из исходной пары — принадлежность требует сверки'
    previous_link = f'<a rel="prev" href="{url(previous)}">← {h(previous["title"])} <small>№ {previous["id"]}</small></a>' if previous else '<span></span>'
    next_link = f'<a rel="next" href="{url(following)}">{h(following["title"])} <small>№ {following["id"]}</small> →</a>' if following else '<span></span>'
    return f'''<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{h(soil['title'])} · {soil['id']} — TerraMind</title>
<meta name="description" content="{h(soil['summary'])}"><link rel="stylesheet" href="/catalog.css"></head>
<body class="soil-detail-page"><header class="detail-topbar"><a class="detail-brand" href="/">TerraMind<span>Почвенный музей</span></a><nav aria-label="Основная навигация"><a href="/#catalog">Каталог</a><a href="/#ai-chat">ИИ-почвовед</a></nav></header>
<main class="detail-main"><a class="back-link" href="/#catalog">← Все образцы</a>
<div class="detail-heading"><p class="catalog-eyebrow">{h(soil['family'])} <span> / {soil['id']}</span></p><h1>{h(soil['title'])}</h1><p>{h(soil['summary'])}</p></div>
<div class="detail-layout"><div class="detail-gallery">
<figure class="monolith-figure"><a href="{h(soil['images']['monolith']['src'])}" target="_blank" rel="noopener" aria-label="Открыть фотографию монолита в полном размере">{photo(soil,'monolith',lazy=False)}</a><figcaption>{h(image_caption)} · нажмите для увеличения</figcaption></figure>
<figure class="label-figure"><a href="{h(soil['images']['label']['src'])}" target="_blank" rel="noopener" aria-label="Открыть музейную табличку в полном размере">{photo(soil,'label')}</a><figcaption>Музейная табличка · нажмите для увеличения</figcaption></figure>
</div><div class="detail-content"><section class="detail-section"><h2>Об образце</h2><p>{h(soil['description'])}</p></section>
{notes}<section class="detail-section"><h2>Данные музейной таблички</h2>{facts(soil['facts'])}</section>{horizons}{scales}
<details class="source-details"><summary>Источник фотографий и описания</summary>
<p>Архив «почвы.zip». {h(soil['pairing'])}</p><p>Табличка: <code>{h(soil['source_files']['label'])}</code><br>Монолит: <code>{h(soil['source_files']['monolith'])}</code></p>
<p>Описание составлено по табличке. Названия мест сохранены в исторической форме. Номер {soil['id']} присвоен в цифровом каталоге и не является музейным инвентарным номером.</p>
<a href="/data/catalog.txt" download>Скачать текст каталога</a></details></div></div>
<nav class="specimen-navigation" aria-label="Соседние образцы">{previous_link}{next_link}</nav>
</main><footer class="detail-footer">TerraMind · Коллекция почвенных монолитов · <a href="/privacy/">Конфиденциальность</a> <a href="/#catalog">Вернуться в каталог ↑</a></footer></body></html>
'''


def transcript(data):
    lines = ['КАТАЛОГ ПОЧВЕННЫХ МОНОЛИТОВ', '', data['source_note'], '',
             'Сопоставление: внутри каждой серии нечётное фото — табличка, следующее чётное — монолит. Независимые инвентарные номера на монолитах не видны. Дубликат 031/041 отмечен отдельно.', '']
    for soil in data['soils']:
        lines += [f'№ {soil["id"]}. {soil["title"]}', f'Страница: {url(soil)}', soil['description'], '']
        lines += [f'{r["label"]}: {r["value"]}' for r in soil['facts']]
        if soil['horizons']:
            lines += ['Горизонты (глубина, см):'] + [f'{r["name"]}: {r["depth_cm"]}' for r in soil['horizons']]
        if soil.get('chart_scales'):
            lines += ['Отметки шкал графиков (не результаты измерений):'] + [f'{r["label"]}: {r["value"]}' for r in soil['chart_scales']]
        lines += ['Примечание: '+n for n in soil['notes']]
        lines += [f'Табличка: {soil["source_files"]["label"]}', f'Монолит: {soil["source_files"]["monolith"]}', '', '—' * 50, '']
    return '\n'.join(lines).rstrip() + '\n'


def outputs():
    data = json.loads((STATIC / 'data/soils.json').read_text(encoding='utf-8'))
    soils = data['soils']
    ids = [s['id'] for s in soils]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'\d{3}', i) for i in ids):
        raise ValueError('Soil IDs must be unique three-digit strings')
    result = {STATIC / 'data/catalog.txt': transcript(data)}
    for i, soil in enumerate(soils):
        result[STATIC / 'soils' / soil['id'] / 'index.html'] = detail(
            soil, soils[i - 1] if i else None, soils[i + 1] if i + 1 < len(soils) else None)
    index = STATIC / 'index.html'
    html = index.read_text(encoding='utf-8')
    if html.count(START) != 1 or html.count(END) != 1:
        raise ValueError('Expected one pair of catalog markers in index.html')
    result[index] = html.split(START)[0] + catalog(soils) + html.split(END)[1]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stale = []
    for path, content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding='utf-8') != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8', newline='\n')
    if stale:
        parser.exit(1, 'Regenerate catalog: ' + ', '.join(stale) + '\n')
    print('Catalog is up to date' if args.check else 'Generated 48 soil pages, catalog cards and text export')


if __name__ == '__main__':
    main()
