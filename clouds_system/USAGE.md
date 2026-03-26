# Руководство по использованию clouds_system

## Установка зависимостей

```
numpy >= 2.4.3
numba >= 0.60.0
```

## Быстрый старт

```python
from clouds_system import CloudSystem, CloudConfig, warmup

# прогрев JIT один раз при старте приложения
warmup()

cs = CloudSystem(CloudConfig(world_seed=42))
cs.next_day()

snapshot = cs.build_snapshot(n_ticks=24)
```

---

## Конфигурация

### `CloudConfig`

Основной объект настройки системы. Все параметры опциональны.

```python
from clouds_system import CloudConfig, LayerConfig

config = CloudConfig(
    world_seed=42,      # int   — сид сессии, определяет всю последовательность дней
    map_width=81,       # int   — ширина карты в клетках
    map_height=40,      # int   — высота карты в клетках
    scale=20.0,         # float — масштаб шума (больше = крупнее облака)
    octaves=6,          # int   — детализация шума (больше = сложнее форма)
    max_shift=20,       # int   — максимальный сдвиг облаков за день (в клетках)
    cloud_thresh=0.55,  # float — базовый порог облачности [0..1]
    layers=[...],       # list[LayerConfig] — слои (по умолчанию: 3 стандартных)
)
```

### `LayerConfig`

Параметры одного высотного слоя. Передаётся в `CloudConfig.layers`.

```python
from clouds_system import LayerConfig

layer = LayerConfig(
    name="низкие",      # str   — название слоя
    scale_mult=0.6,     # float — множитель масштаба шума (< 1 = мельче облака)
    shift_mult=0.5,     # float — множитель скорости ветра (< 1 = медленнее)
    thresh_delta=-0.05, # float — смещение порога облачности от базового
    seed_salt=0x11111111, # int — соль для независимого шума этого слоя
)
```

Стандартные три слоя доступны как `DEFAULT_LAYERS`:

```python
from clouds_system import DEFAULT_LAYERS
```

### Кастомные слои

```python
cs = CloudSystem(CloudConfig(
    world_seed=7,
    layers=[
        LayerConfig("нижний", scale_mult=0.5, shift_mult=0.3, thresh_delta=-0.1, seed_salt=0xAAAA),
        LayerConfig("верхний", scale_mult=2.0, shift_mult=2.0, thresh_delta=0.1,  seed_salt=0xBBBB),
    ]
))
```

---

## Прогрев JIT

Numba компилирует код при первом вызове. Чтобы эта задержка не случилась в момент
первого запроса, вызовите `warmup()` один раз при старте приложения:

```python
from clouds_system import warmup

warmup(scale=20.0, octaves=6)  # параметры должны совпадать с CloudConfig
```

Если `warmup()` не вызван явно — он сработает автоматически при первом `CloudSystem()`,
но заблокирует создание объекта на время компиляции (~1-3 сек).

---

## Жизненный цикл

```python
cs = CloudSystem(CloudConfig(world_seed=42))

# каждый новый день
cs.next_day()
snapshot = cs.build_snapshot(n_ticks=24)

# следующий день
cs.next_day()
snapshot = cs.build_snapshot(n_ticks=24)
```

`next_day()` обязателен перед любым получением данных.

---

## Получение данных

### `build_snapshot(n_ticks)` — основной метод

Возвращает `DaySnapshot` с полным состоянием дня.

```python
snapshot = cs.build_snapshot(n_ticks=24)

snapshot.day               # int — номер текущего дня
snapshot.wind_directions   # dict[str, str] — {"низкие": "SW", "средние": "SW", ...}
snapshot.ticks             # np.ndarray shape (n_ticks, H, W), dtype uint8
```

Работа с `ticks`:

```python
snapshot.ticks[0]          # начало дня → (H, W)
snapshot.ticks[11]         # полдень → (H, W)
snapshot.ticks[23]         # конец дня → (H, W)

snapshot.ticks[5, 10, 20]  # клетка x=20, y=10 в тике 5 → 0 или 1
```

### Низкоуровневые методы

Если нужен контроль над отдельными слоями или моментом времени:

```python
t = 0.5   # момент дня: 0.0 = начало, 1.0 = конец

# карта плотности одного слоя, float32 [0..1]
density = cs.get_layer(layer=0, t=t)

# карты плотности всех слоёв
layers = cs.get_layers(t=t)   # list[np.ndarray]

# булева маска одного слоя
mask = cs.get_layer_mask(layer=0, t=t)   # np.ndarray bool

# булевы маски всех слоёв
masks = cs.get_layer_masks(t=t)   # list[np.ndarray]

# объединённая плотность (максимум по слоям)
combined_density = cs.get(t=t)   # np.ndarray float32

# объединённая маска (хотя бы один слой облачен)
cloudy = cs.is_cloudy(t=t)   # np.ndarray bool
```

---

## Воспроизведение произвольного дня

Сид детерминирован — любой день можно получить без хранения истории:

```python
def get_day(world_seed: int, day: int, n_ticks: int = 24) -> DaySnapshot:
    cs = CloudSystem(CloudConfig(world_seed=world_seed))
    for _ in range(day):
        cs.next_day()
    return cs.build_snapshot(n_ticks)

snapshot = get_day(world_seed=42, day=5)
```

---

## Публичный API

| Имя | Тип | Описание |
|-----|-----|---------|
| `CloudSystem` | class | Основной класс системы облаков |
| `CloudConfig` | dataclass | Конфигурация системы |
| `LayerConfig` | dataclass | Конфигурация одного слоя |
| `DaySnapshot` | dataclass | Результат `build_snapshot()` |
| `DEFAULT_LAYERS` | list[LayerConfig] | Стандартные три слоя |
| `WIND_VECTORS` | dict[str, tuple] | Векторы направлений ветра |
| `WIND_NAMES` | list[str] | Названия направлений ветра |
| `warmup()` | function | Прогрев JIT-компилятора |
