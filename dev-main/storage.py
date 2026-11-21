"""Keep runtime accounts outside tracked source files; preserve an old SQLite DB."""
import os
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile

ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = ROOT / 'data' / 'app.db'
LEGACY_DATABASE = ROOT / 'app.db'


def default_database_path():
    value = os.getenv('TERRAMIND_DB_PATH')
    path = Path(value).expanduser() if value else DEFAULT_DATABASE
    return str(path if path.is_absolute() else ROOT / path)


def prepare_database_path(filename):
    """Copy a legacy database once, with SQLite backup (including WAL changes).

    An existing destination is never replaced. The original file is preserved.
    Called before application startup; only one API process is supported.
    """
    if str(filename) == ':memory:':
        return
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() != DEFAULT_DATABASE.resolve() or destination.exists() or not LEGACY_DATABASE.exists():
        return
    descriptor, temporary = tempfile.mkstemp(prefix='.migration-', suffix='.db', dir=destination.parent)
    os.close(descriptor)
    try:
        source_uri = LEGACY_DATABASE.resolve().as_uri() + '?mode=ro'
        # sqlite3.Connection.__exit__ commits/rolls back, but does not close.
        # Explicitly release both handles before Windows renames or deletes files.
        with closing(sqlite3.connect(source_uri, uri=True)) as source, \
                closing(sqlite3.connect(temporary)) as target:
            source.backup(target)
            if target.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise RuntimeError('Не удалось проверить копию базы данных. Исходная база сохранена.')
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
