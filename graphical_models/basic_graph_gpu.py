from itertools import combinations


class Graph:
    """
    A Graph with a single edge-head style. Used by DAG and UndirectedGraph as a base class
    """
    def __init__(self, nodes_set):
        assert isinstance(nodes_set, set)

        self._graph = dict()
        self.nodes_set = nodes_set
        self.create_empty_graph(self.nodes_set)

    # --- graph initialization functions ------------------------------------------------------------------------------
    # stores the graph as an adjacency list
    def create_empty_graph(self, nodes_set=None):
        if nodes_set is None:
            nodes_set = self.nodes_set
        else:
            assert isinstance(nodes_set, set)

        for node in nodes_set:
            self._graph[node] = set()

    # --- graph query functions ---------------------------------------------------------------------------------------
    def is_connected(self, node_i, node_j):
        if (node_i in self._graph[node_j]) or (node_j in self._graph[node_i]):
            return True
        else:
            return False


    def get_neighbors(self, node_i):
        neighbors = []
        for node_j in (self.nodes_set - {node_i}):
            if self.is_connected(node_i, node_j):
                neighbors.append(node_j)
        return neighbors
