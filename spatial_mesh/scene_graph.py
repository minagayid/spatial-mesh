from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx

from .spaxel import Spaxel
from .space_tensor import SpaceTensor


@dataclass
class SceneNode:
    id: int
    label: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    object_type: str = "unknown"
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    spaxels: List[Spaxel] = field(default_factory=list)


class SpatialSceneGraph:
    def __init__(self):
        self.nodes: Dict[int, SceneNode] = {}
        self.graph = nx.DiGraph()
        self.next_id = 0

    def add_object(self, label, position, size, object_type="unknown", confidence=0.0, metadata=None, spaxels=None):
        nid = self.next_id
        self.next_id += 1
        node = SceneNode(nid, label, position, size, object_type, confidence, metadata or {}, spaxels or [])
        self.nodes[nid] = node
        self.graph.add_node(nid, label=label, type=object_type)
        return nid

    def relate(self, source_id, target_id, relation):
        self.graph.add_edge(source_id, target_id, relation=relation)

    def neighbors(self, node_id, relation=None):
        if relation is None:
            return list(self.graph.successors(node_id))
        out = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if data.get("relation") == relation:
                out.append(target)
        return out
