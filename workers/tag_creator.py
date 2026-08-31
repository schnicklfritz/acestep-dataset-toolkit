"""Structural Tag Creator worker thread."""
from PySide6.QtCore import QThread, Signal

from modules.tag_creator import tag_creator_messages, parse_output


class TagCreatorWorker(QThread):
    finished_ok = Signal(int, str, str)   # sample_index, caption_block, lyrics_block
    failed = Signal(str)

    def __init__(self, sample_index, sample, config, parent=None):
        super().__init__(parent)
        self.sample_index = sample_index
        self.sample = sample
        self.config = config

    def run(self):
        from modules.llm_client import get_client

        name, info, client = get_client(self.config, role="aggregator")
        messages = tag_creator_messages(self.sample)
        resp = client.chat.completions.create(
            model=info.get("model") or "deepseek-chat",
            messages=messages,
            temperature=0.4,
            max_tokens=1400,
        )
        text = (resp.choices[0].message.content or "")
        caption, lyrics = parse_output(text)
        self.finished_ok.emit(self.sample_index, caption, lyrics)