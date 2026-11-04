"""Shared gateway credentials; environment values override the configured .env."""
import os

from src import config


def settings():
    keys = ('VIBEG_API_KEY', 'VIBEG_OPENAI_BASE_URL')
    values = {}
    path = config.GATEWAY_ENV_FILE
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix('export ')
            key, separator, value = line.partition('=')
            if separator and key.strip() in keys:
                values[key.strip()] = value.strip().strip('\"\'')
    values.update({key: os.environ[key] for key in keys if key in os.environ})
    return values
