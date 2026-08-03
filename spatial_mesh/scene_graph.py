from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from .spaxel import Spaxel


class _DirectedGraph:
    """Tiny NetworkX-compatible subset used by the local scene graph."""

    def __init__(self) -> None:
        self._nodes: Dict[int, Dict[str, Any]] = {}
        self._edges: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def add_node(self, node_id: int, **data: Any) -> None:
        self._nodes[node_id] = dict(data)

    def add_edge(self, source: int, target: int, **data: Any) -> None:
        self._edges[(source, target)] = dict(data)

    def successors(self, node_id: int) -> Iterable[int]:
        return [target for source, target in self._edges if source == node_id]

    def out_edges(self, node_id: int, data: bool = False):
        edges = [(source, target, self._edges[(source, target)]) for source, target in self._edges if source == node_id]
        return edges if data else [(source, target) for source, target, _ in edges]

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return len(self._edges)

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()


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
        self.graph = _DirectedGraph()
        self.next_id = 0

    def clear(self) -> None:
        self.nodes.clear()
        self.graph.clear()
        self.next_id = 0

    def add_object(self, label, position, size, object_type="unknown", confidence=0.0, metadata=None, spaxels=None):
        nid = self.next_id
        self.next_id += 1
        node = SceneNode(nid, str(label), tuple(position), tuple(size), str(object_type), float(confidence), metadata or {}, spaxels or [])
        self.nodes[nid] = node
        self.graph.add_node(nid, label=node.label, type=node.object_type)
        return nid

    def _validate_id(self, node_id: int) -> None:
        if not isinstance(node_id, int) or node_id not in self.nodes:
            raise KeyError(f"unknown scene node id: {node_id!r}")

    def relate(self, source_id, target_id, relation):
        self._validate_id(source_id)
        self._validate_id(target_id)
        self.graph.add_edge(source_id, target_id, relation=str(relation))

    def neighbors(self, node_id, relation=None):
        self._validate_id(node_id)
        if relation is None:
            return list(self.graph.successors(node_id))
        return [target for _, target, data in self.graph.out_edges(node_id, data=True) if data.get("relation") == relation]
