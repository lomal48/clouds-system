"""
Cloud system for a session-based game — multi-layer version.

Usage:
    from clouds_system import CloudSystem, CloudConfig, LayerConfig

    cs = CloudSystem(CloudConfig(world_seed=42, map_width=81, map_height=40))
    cs.next_day()

    # плотность по слоям (float32 [0..1]):
    low, mid, high = cs.get_layers(t=0.5)

    # булевые маски по слоям:
    masks = cs.get_layer_masks(t=0.5)

    # объединённая маска (любой слой облачен):
    combined = cs.is_cloudy(t=0.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
from numba import njit, prange


# ── Направления ветра ─────────────────────────────────────────────────────────

WIND_VECTORS: dict[str, tuple[int, int]] = {
    "N":  ( 0, -1), "NE": ( 1, -1), "E":  ( 1,  0), "SE": ( 1,  1),
    "S":  ( 0,  1), "SW": (-1,  1), "W":  (-1,  0), "NW": (-1, -1),
}
WIND_NAMES: list[str] = list(WIND_VECTORS.keys())

# Константа Кнута для хеш-рассеивания seed'а по дням
_DAY_SEED_MULTIPLIER: int = 2654435761
# XOR-маска для отделения seed'а ветра от seed'а шума
_WIND_SEED_MASK: int = 0xDEADBEEF


# ── Конфигурация одного слоя ──────────────────────────────────────────────────

@dataclass
class LayerConfig:
    """
    Параметры одного высотного слоя облаков.

    Attributes
    ----------
    name        : отображаемое название слоя
    scale_mult  : множитель масштаба шума (>1 = крупнее облака)
    shift_mult  : множитель скорости ветра (>1 = быстрее)
    thresh_delta: смещение порога облачности относительно базового
    seed_salt   : уникальная соль для независимого шума слоя
    """
    name:         str
    scale_mult:   float
    shift_mult:   float
    thresh_delta: float
    seed_salt:    int


@dataclass
class DaySnapshot:
    """
    Состояние облаков за один игровой день.

    Attributes
    ----------
    day             : номер игрового дня
    wind_directions : направление ветра по каждому слою {"низкие": "SW", ...}
    ticks           : булева сетка облачности shape (n_ticks, H, W), dtype uint8
                      1 = облачно, 0 = ясно
    """
    day:             int
    wind_directions: dict[str, str]
    ticks:           np.ndarray


DEFAULT_LAYERS: list[LayerConfig] = [
    LayerConfig(name="низкие",  scale_mult=0.6, shift_mult=0.5, thresh_delta=-0.05, seed_salt=0x11111111),
    LayerConfig(name="средние", scale_mult=1.0, shift_mult=1.0, thresh_delta= 0.00, seed_salt=0x22222222),
    LayerConfig(name="высокие", scale_mult=1.7, shift_mult=1.8, thresh_delta= 0.05, seed_salt=0x33333333),
]


# ── Конфигурация системы облаков ──────────────────────────────────────────────

@dataclass
class CloudConfig:
    """
    Параметры системы облаков.

    Attributes
    ----------
    world_seed   : воспроизводимый сид всей сессии
    map_width    : ширина карты в клетках
    map_height   : высота карты в клетках
    scale        : базовый масштаб шума Перлина
    octaves      : количество октав шума
    max_shift    : максимальный сдвиг облаков за день (для базового слоя)
    cloud_thresh : базовый порог облачности [0..1]
    layers       : список конфигураций слоёв
    """
    world_seed:   int               = 0
    map_width:    int               = 81
    map_height:   int               = 40
    scale:        float             = 20.0
    octaves:      int               = 6
    max_shift:    int               = 20
    cloud_thresh: float             = 0.55
    layers:       list[LayerConfig] = field(default_factory=lambda: list(DEFAULT_LAYERS))


# ── Генератор шума Перлина ─────────────────────────────────────────────────────

@njit(parallel=True, fastmath=True, cache=True)
def _perlin_core(width, height, perm, gx, gy, freq, amplitude, out):
    for y in prange(height):
        for x in range(width):
            xf = x * freq;  xi = int(xf) & 255;  xf -= int(xf)
            yf = y * freq;  yi = int(yf) & 255;  yf -= int(yf)
            u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
            v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)
            aa = perm[(perm[xi]     + yi)     & 255]
            ab = perm[(perm[xi]     + yi + 1) & 255]
            ba = perm[(perm[xi + 1] + yi)     & 255]
            bb = perm[(perm[xi + 1] + yi + 1) & 255]
            n00 = gx[aa] * xf       + gy[aa] * yf
            n10 = gx[ba] * (xf - 1) + gy[ba] * yf
            n01 = gx[ab] * xf       + gy[ab] * (yf - 1)
            n11 = gx[bb] * (xf - 1) + gy[bb] * (yf - 1)
            x1  = n00 + u * (n10 - n00)
            x2  = n01 + u * (n11 - n01)
            out[y, x] += (x1 + v * (x2 - x1)) * amplitude


def _generate_noise(width: int, height: int, seed: int,
                    scale: float, octaves: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = np.arange(256, dtype=np.int32)
    rng.shuffle(perm)
    perm = np.tile(perm, 2)
    angles = rng.random(256).astype(np.float32) * 6.2831853
    gx = np.cos(angles)
    gy = np.sin(angles)

    out = np.zeros((height, width), dtype=np.float32)
    amplitude, frequency, max_val = 1.0, 1.0 / scale, 0.0
    for _ in range(octaves):
        _perlin_core(width, height, perm, gx, gy, frequency, amplitude, out)
        max_val += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    out = out / max_val + 0.5
    np.clip(out, 0, 1, out=out)
    return out


# ── Прогрев JIT ───────────────────────────────────────────────────────────────

_WARMED_UP: bool = False


def warmup(scale: float = 20.0, octaves: int = 6) -> None:
    """
    Принудительно запускает JIT-компиляцию Numba с реальными параметрами.

    Вызывать один раз при старте приложения — до создания CloudSystem.

    Parameters
    ----------
    scale   : масштаб шума, который будет использоваться в CloudConfig
    octaves : количество октав, которое будет использоваться в CloudConfig
    """
    global _WARMED_UP
    if not _WARMED_UP:
        _generate_noise(4, 4, seed=0, scale=scale, octaves=octaves)
        _WARMED_UP = True


# ── CloudSystem ───────────────────────────────────────────────────────────────

class CloudSystem:
    """
    Управляет генерацией облаков по высотным слоям.

    Parameters
    ----------
    config : CloudConfig — полная конфигурация системы.
             Передайте кастомный CloudConfig чтобы изменить любые параметры,
             в том числе количество и характеристики слоёв.
    """

    def __init__(self, config: CloudConfig | None = None) -> None:
        self.config: CloudConfig = config if config is not None else CloudConfig()

        self.day: int = 0

        n = len(self.config.layers)
        self._base_maps:      list[np.ndarray | None] = [None] * n
        self._wind_vecs:      list[tuple[int, int]]   = [(0, -1)] * n
        self._pads:           list[int]               = [0] * n
        self.wind_directions: list[str]               = ["N"] * n

        if not _WARMED_UP:
            warmup(scale=self.config.scale, octaves=self.config.octaves)

    # ── свойства ───────────────────────────────────────────────────────────────

    @property
    def n_layers(self) -> int:
        return len(self.config.layers)

    @property
    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.config.layers]

    # ── публичный API ──────────────────────────────────────────────────────────

    def next_day(self) -> None:
        """
        Переход к следующему игровому дню.

        Каждый слой получает независимый шум и скорость ветра.
        С дня 2 каждый слой стартует с конечного положения предыдущего дня.
        """
        cfg = self.config
        self.day += 1
        day_seed = cfg.world_seed + self.day * _DAY_SEED_MULTIPLIER

        rng = np.random.default_rng(day_seed ^ _WIND_SEED_MASK)
        wind_dir = WIND_NAMES[int(rng.integers(len(WIND_NAMES)))]
        wind_vec = WIND_VECTORS[wind_dir]

        for i, layer in enumerate(cfg.layers):
            layer_seed = day_seed + layer.seed_salt
            pad = math.ceil(cfg.max_shift * layer.shift_mult)

            self._pads[i]       = pad
            self._base_maps[i]  = _generate_noise(
                cfg.map_width  + 2 * pad,
                cfg.map_height + 2 * pad,
                seed=layer_seed,
                scale=cfg.scale * layer.scale_mult,
                octaves=cfg.octaves,
            )
            self.wind_directions[i] = wind_dir
            self._wind_vecs[i]      = wind_vec

    def get_layer(self, layer: int, t: float = 0.0) -> np.ndarray:
        """Карта плотности (float32, 0..1) для слоя layer в момент t ∈ [0,1]."""
        if self._base_maps[layer] is None:
            raise RuntimeError("Call next_day() before get_layer()")
        cfg = self.config
        t = float(np.clip(t, 0.0, 1.0))
        shift_m = cfg.layers[layer].shift_mult
        dx, dy = self._wind_vecs[layer]
        pad = self._pads[layer]
        ox = pad + int(round(dx * cfg.max_shift * shift_m * t))
        oy = pad + int(round(dy * cfg.max_shift * shift_m * t))
        return self._base_maps[layer][oy : oy + cfg.map_height, ox : ox + cfg.map_width]

    def get_layers(self, t: float = 0.0) -> list[np.ndarray]:
        """Список карт плотности (float32) для всех слоёв."""
        return [self.get_layer(i, t) for i in range(self.n_layers)]

    def get_layer_mask(self, layer: int, t: float = 0.0) -> np.ndarray:
        """Булевая маска облачности для одного слоя."""
        thresh = self.config.cloud_thresh + self.config.layers[layer].thresh_delta
        return self.get_layer(layer, t) >= thresh

    def get_layer_masks(self, t: float = 0.0) -> list[np.ndarray]:
        """Список булевых масок для всех слоёв."""
        return [self.get_layer_mask(i, t) for i in range(self.n_layers)]

    def get(self, t: float = 0.0) -> np.ndarray:
        """Объединённая карта плотности (максимум по всем слоям)."""
        return np.maximum.reduce(self.get_layers(t))

    def is_cloudy(self, t: float = 0.0) -> np.ndarray:
        """Булевая маска: True если хотя бы один слой облачен."""
        return np.any(np.stack(self.get_layer_masks(t)), axis=0)

    def build_snapshot(self, n_ticks: int = 24) -> DaySnapshot:
        """
        Возвращает состояние облаков за текущий день.

        Parameters
        ----------
        n_ticks : количество временных срезов (равномерно по дню)

        Returns
        -------
        DaySnapshot с полем ticks shape (n_ticks, H, W), dtype uint8
        """
        if self._base_maps[0] is None:
            raise RuntimeError("Call next_day() before build_snapshot()")

        cfg = self.config
        ticks = np.empty((n_ticks, cfg.map_height, cfg.map_width), dtype=np.uint8)
        for tick in range(n_ticks):
            t = tick / (n_ticks - 1) if n_ticks > 1 else 0.0
            ticks[tick] = self.is_cloudy(t).view(np.uint8)

        return DaySnapshot(
            day=self.day,
            wind_directions=dict(zip(self.layer_names, self.wind_directions)),
            ticks=ticks,
        )
