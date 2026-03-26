# clouds-system

Procedural cloud generation system for session-based games.

- 3-layer cloud model (low / mid / high altitude)
- Perlin noise with Numba JIT acceleration
- Fully deterministic — same `world_seed` reproduces the same session
- Returns `DaySnapshot(day, wind_directions, ticks)` — no file I/O

## Installation

```bash
pip install clouds-system
```

## Quick start

```python
from clouds_system import CloudSystem, CloudConfig, warmup

warmup()  # JIT compile once at app startup

cs = CloudSystem(CloudConfig(world_seed=42, map_width=81, map_height=40))
cs.next_day()

snapshot = cs.build_snapshot(n_ticks=24)
# snapshot.day              → int
# snapshot.wind_directions  → {"низкие": "SW", ...}
# snapshot.ticks            → np.ndarray (n_ticks, H, W), uint8
```

See [USAGE.md](clouds_system/USAGE.md) for full API reference and [INTERNALS.md](clouds_system/INTERNALS.md) for implementation details.
