# spatial-mesh

An offline-first spatiotemporal perception core for robotics research. It
represents a scene as an uncertainty-aware 3D occupancy field, fuses synthetic
or recorded observations, groups occupied cells into semantic objects, predicts
motion, and plans a path through the field.

The core runs after download with Python 3.10+ and the standard library only.
It does not connect to hardware, transmit RF/acoustic energy, deploy services,
or make claims that the included ray tracer is a physical sensor model.

## Run locally

```bash
cd spatial-mesh
python demos/demo_headless.py
python -m unittest -v test_spatial_mesh.py
```

The headless demo prints one deterministic JSON record containing occupied
cells, scene objects, motion predictions, and a collision-avoiding path.

Optional visualizations are isolated from the core:

```bash
python -m pip install -e ".[visual]"
python demos/demo_pipeline.py
```

## Minimal API

```python
from spatial_mesh import Observation, SpatialMesh

mesh = SpatialMesh(bounds=((0, 0, 0), (4, 4, 2)), cell_size=0.25)
mesh.replay([
    Observation(
        position=(1.0, 1.0, 0.5),
        occupancy=1.0,
        confidence=0.9,
        modality="depth-recording",
        timestamp=1.0,
        event_id="frame-1-hit",
    )
])
print(mesh.digest())
```

`event_id` makes duplicate observations idempotent. `replay` sorts events by
timestamp and event ID, so the same input history produces the same digest.
Snapshots are versioned and can be restored with
`SpatialMesh.from_snapshot(mesh.snapshot())`.

## Structure

- `spatial_mesh/fusion.py` — observation schema and bounded log-odds fusion
- `spatial_mesh/space_tensor.py` — 3D field, cell channels, replay snapshots
- `spatial_mesh/mesh.py` — high-level offline pipeline facade
- `spatial_mesh/semos.py` — deterministic connected-component clustering and prediction
- `spatial_mesh/scene_graph.py` — validated integer scene-node identities
- `spatial_mesh/navigation.py` — bounded A* with clearance inflation and correct diagonal costs
- `spatial_mesh/raycasting.py` — safe synthetic AABB ray intersections
- `spatial_mesh/world.py` — local synthetic world builder
- `spatial_mesh/sensors.py` — deterministic synthetic sensor sampling
- `spatial_mesh/visualize.py` — optional NumPy/matplotlib rendering
- `demos/demo_headless.py` — no-dependency end-to-end example
- `demos/demo_pipeline.py` — optional dataset and visualization pipeline

World maps use sparse allocation by default: empty geometry remains metadata,
and only observed cells allocate Python objects. `SpaceTensor(..., eager=True)`
is available when a fully materialized grid is required for tensor export.

## Design boundary

The innovation is the unified, replayable and uncertainty-aware wave-field
architecture: multiple observation modalities share the same bounded field,
temporal state, scene graph, and navigation layer. It is an engineering
framework for simulation and recorded-data experiments, not a novelty claim for
acoustic mapping, SLAM, RF sensing, or medical/industrial deployment.
