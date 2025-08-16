from typing import List, Tuple, Dict

import carla
import networkx as nx

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


class CarlaFullCoverageRoutePlanner:
    _GRP_SAMPLING_RESOLUTION: float = 2.0

    def __init__(self, carla_map: carla.Map) -> None:
        self._map: carla.Map = carla_map

        self._grp: GlobalRoutePlanner = GlobalRoutePlanner(self._map, self._GRP_SAMPLING_RESOLUTION)

    def plan(self) -> List[Tuple[carla.Waypoint, RoadOption]]:
        topology = self._map.get_topology()
        waypoint_by_id = self._build_waypoint_by_id(topology)
        graph = self._build_graph(topology)
        semieulerian_graph = self._semieulerize(graph)
        eulerian_path = list(nx.eulerian_path(semieulerian_graph))
        route = []
        for start_wp_id, end_wp_id in eulerian_path:
            start_wp = waypoint_by_id[start_wp_id]
            end_wp = waypoint_by_id[end_wp_id]
            segment = self.grp.trace_route(start_wp.transform.location, end_wp.transform.location)
            route.extend(segment)
        return route

    @property
    def map(self) -> carla.Map:
        return self._map

    @property
    def grp(self) -> GlobalRoutePlanner:
        return self._grp

    def _build_graph(self, topology: List[Tuple[carla.Waypoint, carla.Waypoint]]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for start_wp, end_wp in topology:
            start_wp_loc = start_wp.transform.location
            end_wp_loc = end_wp.transform.location
            distance = start_wp_loc.distance(end_wp_loc)
            graph.add_edge(start_wp.id, end_wp.id, weight=distance)
        return graph

    def _build_waypoint_by_id(self, topology: List[Tuple[carla.Waypoint, carla.Waypoint]]) -> Dict[int, carla.Waypoint]:
        waypoint_by_id = {}
        for start_wp, end_wp in topology:
            waypoint_by_id[start_wp.id] = start_wp
            waypoint_by_id[end_wp.id] = end_wp
        return waypoint_by_id

    @staticmethod
    def _semieulerize(G: nx.DiGraph) -> nx.MultiDiGraph:

        if G.order() == 0:
            raise ValueError("G must not be a null graph")

        if not nx.is_strongly_connected(G):
            raise ValueError("G must be strongly connected")

        if nx.is_semieulerian(G) or nx.is_eulerian(G):
            return nx.MultiDiGraph(G)

        neg = {}
        pos = {}

        for v in sorted(G):
            b = G.out_degree(v) - G.in_degree(v)
            if b < 0:
                neg[v] = -b
            elif b > 0:
                pos[v] = b

        distances = {}
        paths = {}

        for u in neg:
            distances_u, paths_u = nx.single_source_dijkstra(G, source=u, weight="weight")
            for v in pos:
                distances[(u, v)] = distances_u[v]
                paths[(u, v)] = paths_u[v]

        worst_u, worst_v = max(((u, v) for u in neg for v in pos), key=distances.__getitem__)

        neg[worst_u] -= 1
        if neg[worst_u] == 0:
            del neg[worst_u]

        pos[worst_v] -= 1
        if pos[worst_v] == 0:
            del pos[worst_v]

        H = nx.MultiDiGraph(G)

        while neg and pos:
            best_u, best_v = min(((u, v) for u in neg for v in pos), key=distances.__getitem__)

            best_path = paths[(best_u, best_v)]

            for u, v in nx.utils.pairwise(best_path):
                edge_data = G.get_edge_data(u, v)
                H.add_edge(u, v, **edge_data)

            neg[best_u] -= 1
            if neg[best_u] == 0:
                del neg[best_u]

            pos[best_v] -= 1
            if pos[best_v] == 0:
                del pos[best_v]

        return H
