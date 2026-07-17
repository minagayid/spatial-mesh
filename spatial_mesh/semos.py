from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .spaxel import Spaxel
from .space_tensor import SpaceTensor
from .scene_graph import SpatialSceneGraph, SceneNode


_CLUSTER_MAX_POINTS = 1024


class SemanticCluster:
    def __init__(self, label, object_type, position, size, confidence):
        self.label = label
        self.object_type = object_type
        self.position = tuple(float(v) for v in position)
        self.size = tuple(float(v) for v in size)
        self.confidence = float(confidence)
        self.spaxels: List[Spaxel] = []
        self.velocity = (0.0, 0.0, 0.0)
        self.timestamps: List[float] = []

    def add_spaxel(self, spaxel):
        self.spaxels.append(spaxel)
        if spaxel.timestamp:
            self.timestamps.append(spaxel.timestamp)
        vx, vy, vz = spaxel.velocity
        if vx or vy or vz:
            self.velocity = (self.velocity[0] + vx, self.velocity[1] + vy, self.velocity[2] + vz)

    def predict_position(self, seconds=1.0):
        vx, vy, vz = self.velocity
        return (self.position[0] + vx * seconds, self.position[1] + vy * seconds, self.position[2] + vz * seconds)


def _dbscan_1d(values, half_width):
    if not values:
        return []
    centers = []
    spans = []
    start = values[0]
    for val in values[1:]:
        if val - start > half_width * 2:
            centers.append((start + val) / 2.0)
            spans.append((start, val))
            start = val
    centers.append((start + values[-1]) / 2.0)
    spans.append((start, values[-1]))
    return list(zip(centers, spans))


class _Clusterer:
    def __init__(self, voxel_size, axis_cluster_half_width):
        self.voxel_size = float(voxel_size)
        self.axis_half = float(axis_cluster_half_width)

    def cluster(self, occupied):
        x_coords, y_coords, z_coords, entries = [], [], [], []
        for idx, cell in occupied:
            ix, iy, iz = idx
            cx = float(cell.x)
            cy = float(cell.y)
            cz = float(cell.z)
            x_coords.append(cx)
            y_coords.append(cy)
            z_coords.append(cz)
            entries.append((idx, cell, (cx, cy, cz)))
        x_groups = _dbscan_1d(sorted(x_coords), self.axis_half)
        y_groups = _dbscan_1d(sorted(y_coords), self.axis_half)
        z_groups = _dbscan_1d(sorted(z_coords), self.axis_half)
        centroids = [(xr[0], yr[0], zr[0]) for xr in x_groups for yr in y_groups for zr in z_groups]
        clusters = [SemanticCluster(f"cluster-{i}", "unknown", c, (self.axis_half * 2, self.axis_half * 2, self.axis_half * 2), 0.5) for i, c in enumerate(centroids)]
        for idx, cell, pos in entries:
            for cluster in clusters:
                if max(abs(a - b) for a, b in zip(pos, cluster.position)) <= self.axis_half * 1.5:
                    cluster.add_spaxel(cell.to_spaxel())
                    break
        out = []
        for cluster in clusters:
            if not cluster.spaxels:
                continue
            xs = sorted(s.x for s in cluster.spaxels)
            ys = sorted(s.y for s in cluster.spaxels)
            zs = sorted(s.z for s in cluster.spaxels)
            cluster.label = self._label_from(cluster)
            cluster.object_type = cluster.label
            cluster.position = ((xs[0] + xs[-1]) / 2.0, (ys[0] + ys[-1]) / 2.0, (zs[0] + zs[-1]) / 2.0)
            cluster.size = (
                max(xs[-1] - xs[0], self.voxel_size),
                max(ys[-1] - ys[0], self.voxel_size),
                max(zs[-1] - zs[0], self.voxel_size),
            )
            cluster.confidence = min(1.0, 0.4 + 0.02 * len(cluster.spaxels))
            out.append(cluster)
        return out

    def _label_from(self, cluster):
        material_vote = {}
        motion_count = 0
        avg_temp = 0.0
        for s in cluster.spaxels:
            if s.material:
                material_vote[s.material] = material_vote.get(s.material, 0) + 1
            motion_count += s.motion
            avg_temp += s.temperature
        if motion_count / max(len(cluster.spaxels), 1) > 0.35:
            return "human"
        if material_vote:
            mat = max(material_vote, key=material_vote.get)
            if mat in {"wood", "metal", "glass", "plastic", "fabric"}:
                return mat
        if abs(avg_temp) > 2.5:
            return "human_or_animal"
        return "object"


class SpaceOperatingSystem:
    def __init__(self, world_size=2.5):
        if world_size <= 0.0:
            raise ValueError("world_size must be positive.")
        self.world_size = float(world_size)
        self.world_model: Optional[SpaceTensor] = None
        self.scene = SpatialSceneGraph()
        self.clusterer = _Clusterer(0.25, 0.25)
        self.current_objects: List[SceneNode] = []
        self.relations = []
        self.last_frame_objects: List[SceneNode] = []
        self._last_clusters: List[SemanticCluster] = []

    def ingest(self, tensor):
        self.world_model = tensor
        occupied = list(tensor.occupied_cells())
        if not occupied:
            self.current_objects = []
            return self.current_objects
        clusters = self.clusterer.cluster(occupied)
        self._last_clusters = list(clusters)
        scene_nodes: List[SceneNode] = []
        for cluster in clusters:
            node_id = self.scene.add_object(
                cluster.label,
                cluster.position,
                cluster.size,
                object_type=cluster.object_type,
                confidence=cluster.confidence,
                metadata={"velocity": cluster.velocity},
                spaxels=cluster.spaxels,
            )
            scene_nodes.append(SceneNode(node_id, cluster.label, cluster.position, cluster.size, cluster.object_type, cluster.confidence))
        self._infer_relations(clusters, scene_nodes)
        self.last_frame_objects = self.current_objects
        self.current_objects = scene_nodes
        return self.current_objects

    def predict(self, seconds=1.0):
        if not self.current_objects:
            return []
        return [cluster.predict_position(seconds) for cluster in getattr(self, "_last_clusters", [])]

    def _infer_relations(self, clusters, scene_nodes):
        supports = [cluster for cluster in clusters if cluster.object_type in {"table", "shelf"}]
        on_top = [cluster for cluster in clusters if cluster.object_type in {"cup", "book", "object"}]
        for lower in supports:
            lower_id = lower.label
            for upper in on_top:
                if lower is upper:
                    continue
                same_xy = abs(lower.position[0] - upper.position[0]) < lower.size[0] and abs(lower.position[1] - upper.position[1]) < lower.size[1]
                vertical_stack = upper.position[2] > lower.position[2]
                if same_xy and vertical_stack:
                    self.scene.relate(lower_id, upper.label, "ON")
