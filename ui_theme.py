# ui_theme.py
import os
import json

def compile_and_apply_theme(manager):
    """Gentoo Mantra: If you can't configure it, consider it a bug.
    Dynamically compiles and maps style sheets out of a user-editable JSON profile.
    """
    config_path = os.path.join(os.path.dirname(__file__), "theme.json")
    
    if not os.path.exists(config_path):
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(manager.custom_theme, f, indent=4)
        except OSError:
            pass
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                manager.custom_theme.update(json.load(f))
        except Exception as e:
            print(f"Theme parsing compilation exception: {e}")

    base_size = int(12 * manager.custom_theme.get("zoom_factor", 1.0))
    font_fam = manager.custom_theme.get("font_family", "Segoe UI")
    bg = manager.custom_theme.get("bg_color", "#1e1e1e")
    panel = manager.custom_theme.get("panel_bg", "#252526")
    text = manager.custom_theme.get("text_color", "#d4d4d4")
    accent = manager.custom_theme.get("accent_color", "#0e639c")

    style = f"""
        QWidget {{ background-color: {bg}; color: {text}; font-family: '{font_fam}'; font-size: {base_size}px; }}
        QGroupBox, QTableWidget, QTextEdit, QLineEdit, QComboBox, QSpinBox, QScrollArea {{ background-color: {panel}; border: 1px solid #3c3c3c; border-radius: 4px; }}
        QPushButton {{ background-color: {accent}; color: #ffffff; border: none; border-radius: 3px; padding: {int(4 * manager.custom_theme.get('zoom_factor', 1.0))}px {int(10 * manager.custom_theme.get('zoom_factor', 1.0))}px; }}
        QPushButton:hover {{ background-color: #1177bb; }}
        QHeaderView::section {{ background-color: {panel}; color: {text}; padding: 4px; border: 1px solid #333; }}
    """
    manager.setStyleSheet(style)
