import os
import re
import librosa
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPalette
from PySide6.QtWidgets import QWidget, QSizePolicy

class WaveformWidget(QWidget):
    """Renders a downsampled waveform for the selected track, with a playhead."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.min_buckets = None
        self.max_buckets = None
        self.position_frac = 0.0
        self.setMinimumHeight(72)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_audio(self, path):
        self.min_buckets = None
        self.max_buckets = None
        self.position_frac = 0.0
        if path and os.path.exists(path):
            self._load(path)
        self.update()

    def _load(self, path):
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            if len(y) == 0:
                return
            n = len(y)
            buckets = min(1200, max(1, n // 256))
            trimmed = y[: n - (n % buckets)]
            arr = trimmed.reshape(buckets, -1)
            self.min_buckets = arr.min(axis=1)
            self.max_buckets = arr.max(axis=1)
        except Exception:
            self.min_buckets = self.max_buckets = None

    def set_position_frac(self, frac):
        self.position_frac = max(0.0, min(1.0, float(frac)))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().color(QPalette.Window))
        w, h = self.width(), self.height()
        mid = h / 2.0
        if self.min_buckets is None or len(self.min_buckets) < 2:
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignCenter, "Select a track to preview the waveform")
            p.end()
            return
        p.setPen(QPen(QColor(14, 99, 156), 1))
        nb = len(self.min_buckets)
        for i in range(nb):
            x0 = i * w / nb
            x1 = (i + 1) * w / nb
            lo = mid + self.min_buckets[i] * (h * 0.45)
            hi = mid + self.max_buckets[i] * (h * 0.45)
            p.drawLine(int(x0), int(lo), int(x1), int(hi))
        px = int(self.position_frac * w)
        p.setPen(QPen(QColor(255, 82, 82), 2))
        p.drawLine(px, 0, px, h)
        p.end()

class ScatterPlotWidget(QWidget):
    """Renders a 2-D embedding scatter colored by genre, with hover/click."""
    point_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = {}
        self.meta = {}
        self._hover = None
        self.setMinimumSize(420, 300)
        self.setMouseTracking(True)

    def set_data(self, coords, meta):
        self.points = {int(k): (float(x), float(y)) for k, (x, y) in coords.items()}
        self.meta = {int(k): v for k, v in meta.items()}
        self._hover = None
        self.update()

    def _normalize(self):
        xs = [p[0] for p in self.points.values()] or [0.0]
        ys = [p[1] for p in self.points.values()] or [0.0]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        rx = (xmax - xmin) or 1.0
        ry = (ymax - ymin) or 1.0
        w = max(1, self.width() - 24)
        h = max(1, self.height() - 24)
        return {
            k: (12 + (x - xmin) / rx * w, 12 + (ymax - y) / ry * h)
            for k, (x, y) in self.points.items()
        }

    def _find_near(self, pos, tol=12.0):
        best, bd = None, tol
        for k, (x, y) in self._normalize().items():
            d = ((x - pos.x()) ** 2 + (y - pos.y()) ** 2) ** 0.5
            if d < bd:
                best, bd = k, d
        return best

    def mouseMoveEvent(self, e):
        k = self._find_near(e.pos())
        if k is not None and k != self._hover:
            self._hover = k
            self.setToolTip(self.meta.get(k, {}).get("filename", str(k)))
            self.update()

    def mousePressEvent(self, e):
        k = self._find_near(e.pos(), 16)
        if k is not None:
            self.point_clicked.emit(k)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().color(QPalette.Window))
        if not self.points:
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Compute embeddings to see the map (similar songs cluster together)")
            p.end()
            return
        palette = [
            QColor(231, 76, 60), QColor(46, 204, 113), QColor(241, 196, 15),
            QColor(52, 152, 219), QColor(155, 89, 182), QColor(26, 188, 156),
            QColor(243, 156, 18), QColor(127, 140, 141),
        ]
        genre_color = {}
        i = 0
        for m in self.meta.values():
            g = (m.get("genre") or "").strip() or "?"
            if g not in genre_color:
                genre_color[g] = palette[i % len(palette)]
                i += 1
        norm = self._normalize()
        for k, (x, y) in norm.items():
            g = (self.meta.get(k, {}).get("genre") or "").strip() or "?"
            p.setPen(Qt.NoPen)
            p.setBrush(genre_color.get(g, QColor(150, 150, 150)))
            r = 6 if k != self._hover else 8
            p.drawEllipse(int(x) - r, int(y) - r, r * 2, r * 2)
        p.setPen(QColor(150, 150, 150))
        legend = "  •  ".join(f"<{g}>" for g in genre_color)
        p.drawText(8, self.height() - 4, legend)
        p.end()

