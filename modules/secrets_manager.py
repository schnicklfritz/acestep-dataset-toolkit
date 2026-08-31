"""Encrypted secrets storage for API credentials.

Two modes, per the app's "treat every user the same" philosophy:

* **Secure storage** — the OS keyring via the `keyring` package (gnome-keyring /
  Secret Service on Linux) when available; otherwise a Fernet-encrypted file
  (`secrets.enc`) whose key is derived from the machine ID.
* **Session-only** — credentials are sent straight to the API from a popup and
  never persisted (`persist=False`).

No secret is ever written to ``settings.json``.
"""
import base64
import hashlib
import json
import os
from pathlib import Path

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:  # pragma: no cover
    keyring = None
    KEYRING_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    Fernet = None
    CRYPTO_AVAILABLE = False

APP_SERVICE = "ace-step-dataset-toolkit"
SECRETS_FILE = Path(__file__).resolve().parent.parent / "secrets.enc"


def _fernet_key():
    """Derive a 32-byte Fernet key from the machine ID (fallback store only)."""
    machine = ""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            machine = Path(path).read_text(encoding="utf-8").strip()
            break
        except OSError:
            continue
    if not machine:
        machine = os.environ.get("HOSTNAME", "unknown-host")
    digest = hashlib.sha256(f"ace-step-toolkit::{machine}".encode()).digest()
    return base64.urlsafe_b64encode(digest)


def get_secret(key, default=None):
    """Return the stored secret for ``key``, or ``default``."""
    if KEYRING_AVAILABLE:
        try:
            val = keyring.get_password(APP_SERVICE, key)
            if val:
                return val
        except Exception:  # noqa: BLE001
            pass
    return _file_get(key, default)


def set_secret(key, value, persist=True):
    """Store a secret. ``persist=False`` keeps it in memory only (session)."""
    if not persist or not value:
        return
    if KEYRING_AVAILABLE:
        try:
            keyring.set_password(APP_SERVICE, key, value)
            return
        except Exception:  # noqa: BLE001
            pass
    _file_set(key, value)


def delete_secret(key):
    """Remove a stored secret."""
    if KEYRING_AVAILABLE:
        try:
            keyring.delete_password(APP_SERVICE, key)
        except Exception:  # noqa: BLE001
            pass
    _file_delete(key)


# ---------------------------------------------------------------------------
# Fernet-encrypted file fallback
# ---------------------------------------------------------------------------
def _file_get(key, default=None):
    if not CRYPTO_AVAILABLE:
        return default
    try:
        data = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
        token = data.get(key)
        if not token:
            return default
        return Fernet(_fernet_key()).decrypt(token.encode()).decode()
    except Exception:  # noqa: BLE001
        return default


def _file_set(key, value):
    if not CRYPTO_AVAILABLE:
        return
    try:
        data = json.loads(SECRETS_FILE.read_text(encoding="utf-8")) if SECRETS_FILE.exists() else {}
    except Exception:  # noqa: BLE001
        data = {}
    data[key] = Fernet(_fernet_key()).encrypt(value.encode()).decode()
    SECRETS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(SECRETS_FILE, 0o600)
    except OSError:
        pass


def _file_delete(key):
    if not CRYPTO_AVAILABLE:
        return
    try:
        data = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    data.pop(key, None)
    SECRETS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
