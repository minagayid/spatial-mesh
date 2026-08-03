# Spatial-Mesh design notes

## Field update

Each cell stores occupancy, reflectivity, motion, velocity, temperature,
material evidence, timestamp, modality, uncertainty, and an observation count.
The public `OccupancyFusion` primitive updates probability in log-odds space:

```text
l(posterior) = l(prior) + confidence * (1 - uncertainty) * l(measurement)
```

Inputs are clamped to `[0, 1]` and log-odds are capped. The tensor cell keeps
the same bounded policy while retaining the other channels for downstream
clustering and prediction.

`Observation.event_id` is an optional replay key. Repeated events are ignored,
and `SpatialMesh.replay` applies a canonical `(timestamp, event_id)` order.
The implementation does not use wall-clock time.

## Temporal state

Velocity is averaged over contributing spaxels, rather than summed. This makes
the one-step prediction independent of the number of voxels used to represent a
physical object. Missing-frame prediction, track association, and calibrated
sensor timing are intentionally left as later research extensions.

## Scene and navigation safety

Scene relations accept only existing integer node IDs. Ingestion clears the
previous frame before constructing the next scene graph, so stale nodes and
relations cannot accumulate.

A* validates both endpoints, uses axis/2D-diagonal/3D-diagonal costs of
`10/14/17`, rejects diagonal corner cutting, refuses occupied cells, and accepts
optional `robot_radius` inflation plus an uncertainty risk weight. It remains a
grid planner rather than a certified motion planner and must be calibrated and
validated further before physical robot use.

## Simulation boundary

The included world and ray tracer are deterministic synthetic fixtures for
testing data flow. They are not replacements for calibrated ultrasound,
mmWave, Wi-Fi CSI, UWB, depth, or camera drivers. Hardware adapters should
convert recorded/device-specific data into `Observation` values outside this
package and must add their own calibration, privacy, safety, and timing tests.
