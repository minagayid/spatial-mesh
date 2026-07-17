# spatial-mesh

Wave-based spatiotemporal perception mesh for robot spatial intelligence. Treats environments as dynamic volumetric fields of **Space Elements (Spaxels)** built from reflected acoustic or RF waves, instead of camera pixels.

## Run

Runnable on Windows with existing venv deps after `pip install -e .`:

```bash
cd spatial-mesh
python demos/demo_pipeline.py
```

That writes visualizations and `data/spaxel_sets/episodes.csv`.

## Visualizations

- `data/spaxel_sets/world_grid.png` - 3D spaxel occupancy scatter
- `data/spaxel_sets/room_scene.png` - per-object spaxels + occupancy heatmap
- `data/spaxel_sets/navigation.png` - A* path through spaxels
- `data/spaxel_sets/scene_graph.png` - clustered object nodes
- `data/spaxel_sets/scene_predictions.png` - predicted positions
- `data/spaxel_sets/sensor_scans.png` - ray hit samples from 3 origins

## Structure

- `spatial_mesh/spaxel.py` - spaxel data model
- `spatial_mesh/space_tensor.py` - 3D voxel grid + channel tensor
- `spatial_mesh/environment.py` - environment objects
- `spatial_mesh/raycasting.py` - AABB ray intersection
- `spatial_mesh/world.py` - world build + raycast wrapper
- `spatial_mesh/semos.py` - semantic clustering + scene graph
- `spatial_mesh/navigation.py` - A* over occupancy grid
- `spatial_mesh/visualize.py` - matplotlib visualizations
- `spatial_mesh/tracer.py` - sensor/tracer stubs
- `spatial_mesh/sensors.py` - sensor suite
- `demos/demo_pipeline.py` - end-to-end demo + dataset generator
