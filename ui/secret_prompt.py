"""Small credential-prompt dialog with a "save securely" choice.

Two modes per the app's philosophy: the user can save the secret to the
encrypted store, or send it to the API from this popup and never persist it.
"""
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout,
)


class SecretPromptDialog(QDialog):
    def __init__(self, title, label, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(label))

        form = QFormLayout()
        self.input = QLineEdit()
        self.input.setEchoMode(QLineEdit.Password)
        form.addRow("Value:", self.input)
        layout.addLayout(form)

        self.remember = QCheckBox("Save securely to the OS keyring (recommended)")
        self.remember.setChecked(True)
        layout.addWidget(self.remember)

        note = QLabel(
            "Unchecking sends the credential to the API for this session only — "
            "it will never be stored."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(note)

        buttons = QHBoxLayout()
        ok = QPushButton("OK")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        layout.addLayout(buttons)

    def value(self):
        return self.input.text().strip()

    def persist_choice(self):
        return self.remember.isChecked()
