import numpy as np
from itertools import combinations
from .undirected_graph_gpu import UndirectedGraphGPU as UndirectedGraph


class MixedGraph:
    """
    A graph for representing equivalence classes such as CPDAG and PAG
    """
    def __init__(self, nodes_set, edge_mark_types):
        assert isinstance(nodes_set, set)

        self.edge_mark_types = set(edge_mark_types)

        self._graph = dict()
        self.nodes_set = nodes_set
        self.create_empty_graph(self.nodes_set)

    # graph initialization functions ----------------------------------------------------------------------------------
    # used
    def create_empty_graph(self, nodes_set=None):
        if nodes_set is None:
            nodes_set = self.nodes_set
        else:
            assert isinstance(nodes_set, set)

        for node in nodes_set:
            self._graph[node] = dict() # each node is a dictionary of edge-marks/head types
            for head_type in self.edge_mark_types:  # loop over arrow head_types
                self._graph[node][head_type] = set()

    # USED
    def create_complete_graph(self, edge_mark, nodes_set=None):
        if nodes_set is None:
            nodes_set = self.nodes_set
        else:
            assert isinstance(nodes_set, set)

        self.create_empty_graph(nodes_set)  # first, clear all arrow-heads

        for node in nodes_set:
            self._graph[node][edge_mark] = nodes_set - {node}  # connect all nodes into the current node

    # --- graph query functions ---------------------------------------------------------------------------------------

    # used
    def is_connected(self, node_i, node_j):
        """
        Test if two nodes are adjacent in the graph. That is, if they are connected by any edge type.
        :param node_i:
        :param node_j:
        :return: True if the nodes are adjacent; otherwise, False
        """
        assert node_i != node_j

        for (node_p, node_c) in [(node_i, node_j), (node_j, node_i)]:  # switch roles "parent"-"child"
            for edge_mark in self.edge_mark_types:  # test all edge marks
                if node_p in self._graph[node_c][edge_mark]:
                    return True

        return False

    # used
    def is_edge(self, node_i, node_j, edge_mark_at_i, edge_mark_at_j):
        """
        Test the esistance of an edge with the given edge-marks.
        :param node_i:
        :param node_j:
        :param edge_mark_at_i:
        :param edge_mark_at_j:
        :return: True if the specific edge exists; otherwise, False.
        """
        assert (edge_mark_at_i in self.edge_mark_types) and (edge_mark_at_j in self.edge_mark_types)

        if node_j in self._graph[node_i][edge_mark_at_i] and node_i in self._graph[node_j][edge_mark_at_j]:
            return True
        else:
            return False

   

    # used
    def find_adjacent_nodes(self, node_i, pool_nodes=None, edge_type=None):
        """
        Find all the nodes that are connected in/out of node_i.
        :param node_i:
        :param pool_nodes: a set of nodes from which to find the adjacent ones (default: all graph nodes)
        :param edge_type: a tuples: (alpha, beta) defining the allowed connecting edge,
            where alpha is the edge-mark at node_i and beta is the edge-mark at the neighbors.
            Default is None indicating that any edge-mark is allowed.
        :return:
        """
        if edge_type is None:
            connected_nodes = set()
            for edge_mark in self.edge_mark_types:
                connected_nodes.update(self._graph[node_i][edge_mark])
        else:
            mark_origin = edge_type[0]
            mark_neighbor = edge_type[1]
            connected_nodes = set(filter(
                lambda neighbor: node_i in self._graph[neighbor][mark_neighbor],
                self._graph[node_i][mark_origin]
            ))

        if pool_nodes is not None:
            connected_nodes = connected_nodes & pool_nodes
        return connected_nodes
    

    # used
    def get_skeleton_graph(self, en_nodes=None) -> UndirectedGraph:
        if en_nodes is None:
            en_nodes = self.nodes_set

        adj_graph = UndirectedGraph(en_nodes.copy())
        for node_i, node_j in combinations(en_nodes, 2):
            if self.is_connected(node_i, node_j):
                adj_graph.add_edge(node_i, node_j)
        return adj_graph

    # --- graph modification functions --------------------------------------------------------------------------------
    # used
    def delete_edge(self, node_i, node_j):
        for edge_mark in self.edge_mark_types:  # loop through all edge marks
            self._graph[node_i][edge_mark].discard(node_j)
            self._graph[node_j][edge_mark].discard(node_i)

    # used
    def replace_edge_mark(self, node_source, node_target, requested_edge_mark):
        assert requested_edge_mark in self.edge_mark_types

        # remove any edge-mark
        for edge_mark in self.edge_mark_types:
            self._graph[node_target][edge_mark].discard(node_source)

        # set requested edge-mark
        self._graph[node_target][requested_edge_mark].add(node_source)

    # used
    def reset_orientations(self, default_mark, nodes_set=None):
        """
        Reset all orientations, e.g., convert all edges into o--o edges, where "o" is the default edge-mark
        :param default_mark: an edge-mark to place the instead of the existing edge_marks
        :param nodes_set: Only edges between pairs of nodes from this set will be converted (default: all edges)
        :return:
        """
        assert default_mark in self.edge_mark_types
        if nodes_set is None:
            nodes_set = self.nodes_set

        for (node_x, node_y) in combinations(nodes_set, 2):
            if self.is_connected(node_x, node_y):
                self.replace_edge_mark(node_x, node_y, default_mark)
                self.replace_edge_mark(node_y, node_x, default_mark)

    def add_edge(self, node_i, node_j, edge_mark_at_i, edge_mark_at_j):
        """
        Add an edge with the requested edge-marks.
        :param node_i:
        :param node_j:
        :param edge_mark_at_i:
        :param edge_mark_at_j:
        :return:
        """

        assert not self.is_connected(node_i, node_j)  # edge already exists
        assert (edge_mark_at_i in self.edge_mark_types) and (edge_mark_at_j in self.edge_mark_types)

        self._graph[node_i][edge_mark_at_i].add(node_j)
        self._graph[node_j][edge_mark_at_j].add(node_i)


    # --- plotting tools ----------------------------------------------------------------------------------------------
    def __str__(self):
        text_print = 'Edge-marks on the graph edges:\n'
        for node in self.nodes_set:
            for edge_mark in self.edge_mark_types:
                if len(self._graph[node][edge_mark]) > 0:
                    text_print += ('Edges: ' + str(node) + ' ' + edge_mark + '*' +
                                   ' ' + str(self._graph[node][edge_mark]) + '\n')
        return text_print
