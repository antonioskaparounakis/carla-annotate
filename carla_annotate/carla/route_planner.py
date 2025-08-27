from typing import List, Tuple, Dict, Optional

import carla
import networkx as nx

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


class RoutePlanner:
    GRP_SAMPLING_RESOLUTION: float = 2.0

    def __init__(self, map: carla.Map):
        self._map = map
        self.grp = GlobalRoutePlanner(self._map, self.GRP_SAMPLING_RESOLUTION)
        topology = self._map.get_topology()
        self._graph = self._build_graph(topology)
        self._id_to_waypoint = self._build_id_to_waypoint(topology)

    def plan(self, strategy: str) -> List[Tuple[carla.Waypoint, RoadOption]]:
        if strategy == "full_coverage":
            return self._plan_full_coverage()
        raise ValueError(f"invalid strategy: {strategy!r}")

    def _plan_full_coverage(self) -> List[Tuple[carla.Waypoint, RoadOption]]:
        semieulerian_graph = self._semieulerize_min_cost_flow(self._graph)
        eulerian_path = list(nx.eulerian_path(semieulerian_graph))
        return self._path_to_route(eulerian_path)

    @staticmethod
    def _build_graph(topology: List[Tuple[carla.Waypoint, carla.Waypoint]]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for entry_wp, exit_wp in topology:
            distance = entry_wp.transform.location.distance(exit_wp.transform.location)
            graph.add_edge(entry_wp.id, exit_wp.id, weight=distance)
        return graph

    @staticmethod
    def _build_id_to_waypoint(topology: List[Tuple[carla.Waypoint, carla.Waypoint]]) -> Dict[int, carla.Waypoint]:
        id_to_waypoint = {}
        for entry_wp, exit_wp in topology:
            id_to_waypoint[entry_wp.id] = entry_wp
            id_to_waypoint[exit_wp.id] = exit_wp
        return id_to_waypoint

    def _path_to_route(self, path: List[Tuple[int, int]]) -> List[Tuple[carla.Waypoint, RoadOption]]:
        route = []
        for entry_wp_id, exit_wp_id in path:
            entry_wp = self._id_to_waypoint[entry_wp_id]
            exit_wp = self._id_to_waypoint[exit_wp_id]
            segment = self.grp.trace_route(entry_wp.transform.location, exit_wp.transform.location)
            if segment:
                route.extend(segment)
        return route

    @staticmethod
    def _semieulerize_greedy(G: nx.DiGraph) -> nx.MultiDiGraph:
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

    @staticmethod
    def _semieulerize_min_cost_flow(G: nx.DiGraph) -> nx.MultiDiGraph:
        """
        Semi-eulerize a strongly connected directed graph by duplicating the shortest paths
        with minimum total added weight, using a min-cost flow reduction.

        Returns a MultiDiGraph that is semi-eulerian (exactly two nodes have imbalance ±1).
        The actual open Euler trail can be computed elsewhere.
        """
        # ---------- 0) Preconditions (same intent as your code) ----------
        if G.order() == 0:
            raise ValueError("G must not be a null graph")

        if not nx.is_strongly_connected(G):
            raise ValueError("G must be strongly connected")

        if nx.is_semieulerian(G) or nx.is_eulerian(G):
            return nx.MultiDiGraph(G)

        # ---------- 1) Build imbalance buckets (same naming) ----------
        neg: Dict[str, int] = {}  # needs extra outgoing (indeg > outdeg) → supply units
        pos: Dict[str, int] = {}  # needs extra incoming (outdeg > indeg) → demand units

        for v in sorted(G):
            b = G.out_degree(v) - G.in_degree(v)
            if b < 0:
                neg[v] = -b
            elif b > 0:
                pos[v] = b

        # Defensive: total supply must equal total demand
        total_units = sum(neg.values())
        if total_units != sum(pos.values()):
            raise ValueError("Total imbalance must match on both sides")

        # ---------- 2) All pairs costs for the imbalance bipartite graph ----------
        # Reuse your structure: precompute distances/paths only from neg → pos.
        distances: Dict[Tuple[str, str], float] = {}
        paths: Dict[Tuple[str, str], List[str]] = {}

        for u in neg:
            dist_u, path_u = nx.single_source_dijkstra(G, source=u, weight="weight")
            for v in pos:
                distances[(u, v)] = dist_u[v]
                paths[(u, v)] = path_u[v]

        # ---------- 3) Tiny helper: solve min-cost flow on residual counts ----------
        def _solve_residual_mcf(neg_counts: Dict[str, int],
                                pos_counts: Dict[str, int]) -> Tuple[int, Dict[str, Dict[str, int]]]:
            """
            Build a bipartite flow network (neg → pos) with arc costs equal to
            shortest-path distances, then run network_simplex.
            Returns (cost, flow_Dict), where flow_Dict[u][v] = integer units sent.
            """
            if not neg_counts and not pos_counts:
                return 0, {}

            F = nx.DiGraph()
            # network_simplex convention: node 'demand' > 0 consumes, < 0 supplies
            for u_, k_ in neg_counts.items():
                F.add_node(u_, demand=-k_)  # supply k units
            for v_, k_ in pos_counts.items():
                F.add_node(v_, demand=+k_)  # demand k units

            BIG = sum(pos_counts.values()) or 1  # sufficient capacity on arcs
            for u_ in neg_counts:
                for v_ in pos_counts:
                    F.add_edge(u_, v_, weight=distances[(u_, v_)], capacity=BIG)

            cost, flow = nx.network_simplex(F)
            return cost, flow  # cost is int if weights are ints

        # ---------- 4) Choose the optimal (end, start) pair to leave unmatched ----------
        # We must leave exactly one unit unmatched: t ∈ neg (trail END), s ∈ pos (trail START).
        best_pair: Optional[Tuple[str, str]] = None
        best_total_cost = float("inf")

        for t in neg:  # t = node with deficit-out (will end here)
            if neg[t] == 0:
                continue
            for s in pos:  # s = node with deficit-in (will start here)
                if pos[s] == 0:
                    continue
                # Remove 1 unit from both sides to encode "semi" (open trail)
                neg2 = neg.copy()
                pos2 = pos.copy()
                neg2[t] -= 1
                if neg2[t] == 0:
                    del neg2[t]
                pos2[s] -= 1
                if pos2[s] == 0:
                    del pos2[s]

                residual_cost, _ = _solve_residual_mcf(neg2, pos2)
                if residual_cost < best_total_cost:
                    best_total_cost = residual_cost
                    best_pair = (t, s)

        assert best_pair is not None, "No feasible semi-eulerization pairing found"
        t_end, s_start = best_pair  # (trail end, trail start)

        # ---------- 5) Solve MCF once more for the chosen start/end, get actual flows ----------
        neg_final = neg.copy()
        pos_final = pos.copy()
        neg_final[t_end] -= 1
        if neg_final[t_end] == 0:
            del neg_final[t_end]
        pos_final[s_start] -= 1
        if pos_final[s_start] == 0:
            del pos_final[s_start]

        _, flow_Dict = _solve_residual_mcf(neg_final, pos_final)

        # ---------- 6) Materialize duplicates along chosen shortest paths ----------
        H = nx.MultiDiGraph(G)
        for u, nbrs in flow_Dict.items():
            for v, f in nbrs.items():
                if f <= 0:
                    continue
                path = paths[(u, v)]
                for _ in range(f):
                    for a, b in nx.utils.pairwise(path):
                        edge_data = G.get_edge_data(a, b)
                        # copy attributes defensively to avoid mutating G's Dict
                        H.add_edge(a, b, **dict(edge_data))

        # Optional sanity: result is semi-eulerian (open trail exists)
        # (keep both checks; some NetworkX versions differ on DiGraph support)
        assert any((H.out_degree(x) - H.in_degree(x)) == +1 for x in H), "No +1 imbalance"
        assert any((H.out_degree(x) - H.in_degree(x)) == -1 for x in H), "No -1 imbalance"
        # If your codebase uses this check, and it works for DiGraph/MultiDiGraph, keep it:
        # assert nx.is_semieulerian(H)

        return H
