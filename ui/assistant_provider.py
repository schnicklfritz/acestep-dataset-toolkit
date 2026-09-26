"""Provider picker shown in the Assistant panel.

One row: which LLM answers, whether it has a key, and a link to get a free
key. It edits the SAME config keys as Settings > LLM Provider
(``llm_provider_assistant`` + the provider's key field), so the two never
disagree. Default is Groq's free tier (see modules/llm_client.py).
"""
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QInputDialog, QLabel, QLineEdit, QPushButton

from modules.llm_client import PROVIDERS, provider_info, provider_key_present
from modules.wheel_guard import GuardedComboBox as QComboBox

ASSISTANT_KEYS = ("llm_provider_assistant",)


def build_provider_row(manager):
    row = QHBoxLayout()
    row.setSpacing(6)
    manager.assistant_provider_combo = QComboBox()
    for name, info in PROVIDERS.items():
        manager.assistant_provider_combo.addItem(info["label"], name)
    manager.assistant_provider_combo.setToolTip(
        "Which LLM answers here. Groq's free tier is the default; the same "
        "choice is under ⚙ Settings → LLM Provider → Assistant."
    )
    manager.assistant_key_btn = QPushButton("Add key")
    manager.assistant_key_btn.clicked.connect(lambda: _ask_key(manager))
    manager.assistant_free_link = QPushButton("Get a free key")
    manager.assistant_free_link.clicked.connect(lambda: _open_signup(manager))
    manager.assistant_provider_state = QLabel("")
    manager.assistant_provider_state.setProperty("muted", True)
    row.addWidget(QLabel("Model:"))
    row.addWidget(manager.assistant_provider_combo, 1)
    row.addWidget(manager.assistant_key_btn)
    row.addWidget(manager.assistant_free_link)

    name, _info = provider_info(manager.config, role="assistant")
    idx = manager.assistant_provider_combo.findData(name)
    manager.assistant_provider_combo.setCurrentIndex(max(0, idx))
    manager.assistant_provider_combo.currentIndexChanged.connect(lambda _i: _on_change(manager))
    refresh(manager)
    return row, manager.assistant_provider_state


def refresh(manager):
    name = manager.assistant_provider_combo.currentData()
    info = PROVIDERS[name]
    has_key = provider_key_present(manager.config, provider=name)
    manager.assistant_key_btn.setText("Change key" if has_key else "Add key")
    manager.assistant_key_btn.setProperty("role", "" if has_key else "primary")
    manager.assistant_free_link.setVisible(bool(info.get("free") and info.get("signup_url")) and not has_key)
    _, resolved = provider_info(manager.config, provider=name, role="assistant")
    model = resolved.get("model") or "(set in Settings)"
    state = "ready" if has_key else "needs a key"
    manager.assistant_provider_state.setText(f"{info['label']} · {model} · {state}")
    from ui.themes import repolish

    repolish(manager.assistant_key_btn)


def _on_change(manager):
    from modules.config_store import save_plain_keys

    manager.config["llm_provider_assistant"] = manager.assistant_provider_combo.currentData()
    save_plain_keys(manager.config, ASSISTANT_KEYS)
    # keep the Settings dialog's per-role picker in sync
    combo = getattr(manager, "role_provider_combo", {}).get("assistant")
    if combo is not None:
        combo.setCurrentText(manager.config["llm_provider_assistant"])
    refresh(manager)


def _ask_key(manager):
    from modules.config_store import save_config

    name = manager.assistant_provider_combo.currentData()
    info = PROVIDERS[name]
    if name == "local":
        manager.settings_action.trigger()
        return
    key, ok = QInputDialog.getText(
        manager, f"{info['label']} API key",
        f"Paste your {info['label']} API key. It is stored encrypted, never in settings.json.",
        QLineEdit.Password,
    )
    if not (ok and key.strip()):
        return
    manager.config[info["key"]] = key.strip()
    try:
        save_config(manager.config, remember=manager._remembered_secret_keys() | {info["key"]})
    except Exception as e:  # noqa: BLE001 -- the key still works this session
        manager.assistant_status.setText(f"Key kept for this session only ({e}).")
    refresh(manager)


def _open_signup(manager):
    url = PROVIDERS[manager.assistant_provider_combo.currentData()].get("signup_url")
    if url:
        QDesktopServices.openUrl(QUrl(url))
