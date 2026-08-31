"""Config persistence: non-secret settings in settings.json, secrets encrypted.

Routing:
* non-secret keys (URLs, preferences, usernames) -> ``settings.json`` (gitignored)
* secret keys (``config.SECRET_KEYS``) -> encrypted store (keyring / secrets.enc)
"""
import json

from config import SECRET_KEYS, SETTINGS_PATH
from modules.secrets_manager import delete_secret, get_secret, set_secret


def load_config(defaults):
    """Build a config dict: defaults + settings.json + encrypted secrets."""
    cfg = dict(defaults)
    try:
        if SETTINGS_PATH.exists():
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            for key in defaults:
                if key in saved:
                    cfg[key] = saved[key]
    except Exception:  # noqa: BLE001
        pass
    for key in SECRET_KEYS:
        val = get_secret(key)
        if val:
            cfg[key] = val
    return cfg


def save_config(cfg, remember=None):
    """Persist cfg: plain keys to settings.json, secrets to the encrypted store.

    ``remember`` optionally restricts which secret keys get persisted. When
    ``None`` every secret is persisted (backwards-compatible default). A secret
    that is *not* in ``remember`` is kept in memory for the session only and is
    removed from the store (security-minded behavior).
    """
    plain = {k: v for k, v in cfg.items() if k not in SECRET_KEYS}
    try:
        SETTINGS_PATH.write_text(json.dumps(plain, indent=2), encoding="utf-8")
    except OSError:
        pass
    if remember is None:
        remember = set(SECRET_KEYS)
    remember = set(remember)
    for key in SECRET_KEYS:
        val = cfg.get(key)
        if key in remember and val:
            set_secret(key, str(val))
        elif key not in remember:
            # Session-only: make sure nothing lingers in the encrypted store.
            delete_secret(key)
