"""Настройки новостного модуля. Всё, что можно менять, задаётся здесь или в .env."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_DATA = PROJECT_ROOT / "user_data"

# Ленты проверены на доступность с этого компьютера. Недоступная лента
# пропускается с предупреждением и не останавливает модуль.
DEFAULT_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "The Block": "https://www.theblock.co/rss.xml",
    "Decrypt": "https://decrypt.co/feed",
}

# Модели, для которых API принимает параметр fallbacks (резервная модель при отказе).
MODELS_WITH_FALLBACKS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2"}


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Читает KEY=VALUE из .env, не перезаписывая уже заданные переменные окружения."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    db_path: Path = USER_DATA / "news.sqlite"
    state_path: Path = USER_DATA / "news_state.json"
    feeds: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FEEDS))
    stake_currency: str = "USDT"
    assets: list[str] = field(default_factory=lambda: ["BTC", "ETH"])

    # auto: есть ключ API -> API, иначе Claude Code по подписке. Можно задать api или cli.
    backend: str = "auto"
    # Пусто — модель по умолчанию для выбранного способа: "opus" для Claude Code
    # (последний Opus, который знает установленная версия), claude-opus-5-5 для API.
    model: str = ""
    effort: str = "low"
    cli_timeout: float = 300.0
    batch_size: int = 15
    # Новости старше этого возраста не оцениваются: вето по ним уже бесполезно.
    max_age_hours: int = 12
    # Потолок длительности одного запрета.
    max_veto_hours: int = 48
    http_timeout: float = 20.0


def load_settings() -> Settings:
    load_dotenv()
    settings = Settings()
    settings.backend = os.environ.get("NEWS_BACKEND", settings.backend).lower()
    settings.model = os.environ.get("NEWS_MODEL", settings.model)
    settings.effort = os.environ.get("NEWS_EFFORT", settings.effort)

    # Список монет берём из конфига бота, чтобы он не расходился со стратегией.
    config_path = USER_DATA / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        settings.stake_currency = config.get("stake_currency", settings.stake_currency)
        pairs = config.get("exchange", {}).get("pair_whitelist", [])
        assets = [p.split("/")[0] for p in pairs if "/" in p]
        if assets:
            settings.assets = assets
    return settings
