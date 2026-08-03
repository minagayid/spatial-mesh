from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
        self._velocity_sum = [0.0, 0.0, 0.0]

    def add_spaxel(self, spaxel):
        if len(self.spaxels) >= _CLUSTER_MAX_POINTS:
            return
        self.spaxels.append(spaxel)
        if spaxel.timestamp:
            self.timestamps.append(spaxel.timestamp)
        for axis, value in enumerate(spaxel.velocity):
            self._velocity_sum[axis] += value
        count = float(len(self.spaxels))
        self.velocity = tuple(value / count for value in self._velocity_sum)

    def predict_position(self, seconds=1.0):
        seconds = float(seconds)
        vx, vy, vz = self.velocity
        return (self.position[0] + vx * seconds, self.position[1] + vy * seconds, self.position[2] + vz * seconds)


class _Clusterer:
    def __init__(self, voxel_size, axis_cluster_half_width):
        self.voxel_size = float(voxel_size)
        self.axis_half = float(axis_cluster_half_width)

    def cluster(self, occupied):
        entries = list(occupied)
        if not entries:
            return []
        indexed = {idx: (idx, cell) for idx, cell in entries}
        groups: List[List[Tuple[Tuple[int, int, int], object]]] = []
        by_object: Dict[int, List[Tuple[Tuple[int, int, int], object]]] = {}
        unknown: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], object]] = {}
        for entry in entries:
            idx, cell = entry
            if cell.object_id:
                by_object.setdefault(cell.object_id, []).append(entry)
            else:
                unknown[idx] = entry
        groups.extend(by_object[key] for key in sorted(by_object))

        # Connected components prevent the old Cartesian-product cluster
        # explosion when objects share x/y/z projections.
        while unknown:
            start = next(iter(unknown))
            queue = [start]
            group = []
            del unknown[start]
            while queue:
                current = queue.pop()
                group.append(indexed[current])
                cx, cy, cz = current
                adjacent = [idx for idx in unknown if max(abs(idx[0] - cx), abs(idx[1] - cy), abs(idx[2] - cz)) <= 1]
                for idx in adjacent:
                    del unknown[idx]
                    queue.append(idx)
            groups.append(group)

        clusters = []
        for index, group in enumerate(groups):
            points = [cell.to_spaxel() for _, cell in group[:_CLUSTER_MAX_POINTS]]
            if not points:
                continue
            xs = [s.x for s in points]
            ys = [s.y for s in points]
            zs = [s.z for s in points]
            cluster = SemanticCluster(
                f"cluster-{index}",
                "unknown",
                ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0),
                (max(max(xs) - min(xs), self.voxel_size), max(max(ys) - min(ys), self.voxel_size), max(max(zs) - min(zs), self.voxel_size)),
                min(1.0, 0.4 + 0.02 * len(points)),
            )
            for point in points:
                cluster.add_spaxel(point)
            cluster.label = self._label_from(cluster)
            cluster.object_type = cluster.label
            clusters.append(cluster)
        return clusters

    def _label_from(self, cluster):
        type_vote: Dict[str, int] = {}
        material_vote: Dict[str, int] = {}
        motion = 0.0
        temperature = 0.0
        for spaxel in cluster.spaxels:
            if spaxel.object_type:
                type_vote[spaxel.object_type] = type_vote.get(spaxel.object_type, 0) + 1
            if spaxel.material:
                material_vote[spaxel.material] = material_vote.get(spaxel.material, 0) + 1
            motion += spaxel.motion
            temperature += spaxel.temperature
        if type_vote:
            return max(type_vote, key=type_vote.get)
        if motion / max(len(cluster.spaxels), 1) > 0.35:
            return "human"
        if material_vote:
            return max(material_vote, key=material_vote.get)
        if abs(temperature) / max(len(cluster.spaxels), 1) > 2.5:
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
        self.last_frame_objects = list(self.current_objects)
        self.scene.clear()
        occupied = list(tensor.occupied_cells())
        if not occupied:
            self._last_clusters = []
            self.current_objects = []
            return self.current_objects
        clusters = self.clusterer.cluster(occupied)
        self._last_clusters = list(clusters)
        scene_nodes: List[SceneNode] = []
        cluster_node_ids = {}
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
            cluster_node_ids[id(cluster)] = node_id
            scene_nodes.append(self.scene.nodes[node_id])
        self._infer_relations(clusters, cluster_node_ids)
        self.current_objects = scene_nodes
        return self.current_objects

    def predict(self, seconds=1.0):
        return [cluster.predict_position(seconds) for cluster in self._last_clusters]

    def _infer_relations(self, clusters, cluster_node_ids):
        supports = [cluster for cluster in clusters if cluster.object_type in {"table", "shelf"}]
        on_top = [cluster for cluster in clusters if cluster.object_type in {"cup", "book", "object"}]
        for lower in supports:
            for upper in on_top:
                if lower is upper:
                    continue
                same_xy = abs(lower.position[0] - upper.position[0]) < lower.size[0] and abs(lower.position[1] - upper.position[1]) < lower.size[1]
                vertical_stack = upper.position[2] > lower.position[2]
                if same_xy and vertical_stack:
                    self.scene.relate(cluster_node_ids[id(lower)], cluster_node_ids[id(upper)], "ON")
