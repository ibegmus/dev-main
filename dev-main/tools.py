import db


def format_monolith(m):
    lines = [f"Монолит: {m['name']} (id {m['id']})"]
    fields = [
        ("Регион", m["region"]),
        ("Год поступления", m["year"]),
        ("Тип почвы", m["soil_type"]),
        ("Гумус, %", m["humus"]),
        ("pH", m["ph"]),
        ("Описание", m["description"]),
    ]
    for label, value in fields:
        lines.append(f"  {label}: {value if value is not None else 'нет данных'}")
    return "\n".join(lines)


def tool_find(query: str):
    found = db.find_monoliths(query)
    if not found:
        return ""
    return "\n\n".join(format_monolith(m) for m in found)


def tool_compare(a: str, b: str):
    first = db.find_monoliths(a)
    second = db.find_monoliths(b)
    if not first or not second:
        return ""
    return format_monolith(first[0]) + "\n\n" + format_monolith(second[0])


def tool_list():
    items = db.list_monoliths()
    if not items:
        return ""
    names = "\n".join(f"- {m['name']} ({m['region']})" for m in items)
    return f"В каталоге монолитов: {len(items)}\n{names}"


def run_tool(name: str, args: dict):
    """Возвращает текст результата или пустую строку, если ничего не нашлось."""
    if name == "find":
        return tool_find(str(args.get("query", "")))
    if name == "compare":
        return tool_compare(str(args.get("a", "")), str(args.get("b", "")))
    if name == "list":
        return tool_list()
    return ""