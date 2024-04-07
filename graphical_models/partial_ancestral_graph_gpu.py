from causal_discovery_utils import constraint_based_gpu as constraint_based
from .basic_equivalance_class_graph_gpu import MixedGraph
from .partially_dag_gpu import gpuPDAG as PDAG
from . import arrow_head_types as Mark
from itertools import combinations
import numpy as np

class gpuPAG(MixedGraph):
    """
    Partial Ancestral Graph. It has three arrow-head/edge-mark types: 'circle', 'undirected', and 'directed',
    and six edge types: o--o, o---, o-->, --->, <-->, and ----
    """
    def __init__(self, nodes_set):
        super().__init__(nodes_set, [Mark.Circle, Mark.Directed, Mark.Tail])
        self.sepset = constraint_based.SeparationSet(nodes_set)
        self.visible_edges = None  # a set of visible edges, where each element is a tuple: (parent, child)
        self.orientation_rules = {
            1: self.orient_by_rule_1,
            2: self.orient_by_rule_2,
            3: self.orient_by_rule_3,
            4: self.orient_by_rule_4,
            5: self.orient_by_rule_5,  # when selection bias may be present
            6: self.orient_by_rule_6,  # when selection bias may be present
            7: self.orient_by_rule_7,  # when selection bias may be present
            8: self.orient_by_rule_8,  # required for tail-completeness
            9: self.orient_by_rule_9,  # required for tail-completeness
            10: self.orient_by_rule_10,  # required for tail-completeness
        }

    def init_from_adj_mat(self, adj_mat: np.ndarray, nodes_order: list = None):
        """
        creates a PAG from a given adjacency matrix.
        :param adj_mat: a square numpy matrix.
        an edge a --* b has the following coding:
        adj_mat[a,b] = 0  implies   a   b    (no edge)
        adj_mat[a,b] = 1  implies   a --o b  (Circle marker on node b)
        adj_mat[a,b] = 2  implies   a --> b  (Arrowhead marker on node b)
        adj_mat[a,b] = 3  implies   a --- b  (Tail marker on node b)

        :param nodes_order: nodes ids. if set to None (default) then using sorted(nodes_set) as nodes ids.
        :return:
        """
        assert adj_mat.ndim == 2
        assert adj_mat.shape[0] == adj_mat.shape[1]
        assert np.sum(adj_mat < 0) == 0 and np.sum(adj_mat > 3) == 0

        num_vars = adj_mat.shape[0]
        if nodes_order is not None:
            assert isinstance(nodes_order, list)
            assert num_vars == len(nodes_order)
        else:
            nodes_order = sorted(self.nodes_set)

        self.create_empty_graph()  # delete all pre-existing edges

        # conversion mapping
        arrow_type_map = dict()
        arrow_type_map[0] = 0
        arrow_type_map[1] = Mark.Circle
        arrow_type_map[2] = Mark.Directed
        arrow_type_map[3] = Mark.Tail

        for node_i in range(num_vars):
            for node_j in range(num_vars):
                if adj_mat[node_i, node_j] > 0:
                    arrow_type = arrow_type_map[adj_mat[node_i, node_j]]  # edge mark node_i ---[*] node_j
                    self._graph[nodes_order[node_j]][arrow_type].add(nodes_order[node_i])  # add to node_j edge mark [*]

    def get_adj_mat(self):
        """
        converts a PAG to an adjacency matrix with the following coding:
         0: No edge
         1: Circle
         2: Arrowhead
         3: Tail
        :return: a square numpy matrix format.
        """
        num_vars = len(self.nodes_set)
        adj_mat = np.zeros((num_vars, num_vars), dtype=int)
        node_index_map = {node: i for i, node in enumerate(sorted(self.nodes_set))}

        # convert adjacency to PAG
        arrow_type_map = dict()
        arrow_type_map[Mark.Circle] = 1
        arrow_type_map[Mark.Directed] = 2
        arrow_type_map[Mark.Tail] = 3

        for node in self._graph:
            for edge_mark in self.edge_mark_types:
                for node_p in self._graph[node][edge_mark]:
                    adj_mat[node_index_map[node_p]][node_index_map[node]] = arrow_type_map[edge_mark]

        return adj_mat

    # def fan_in(self, target_node):
    #     """
    #     Return the number of arrow heads <--* and o--* into a node. Do not count tails (undirected)
    #     :param target_node: a node
    #     :return: Fan-in of node target_node
    #     """
    #     return len(self._graph[target_node][Mark.Directed]) + len(self._graph[target_node][Mark.Circle])

    def is_collider(self, node_middle, node_x, node_y):
        """
        Test if X *--> middle-node <--* Y, that is, test if middle-node is a collider.
        :param node_middle:
        :param node_x:
        :param node_y:
        :return: True if the middle node is a collider
        """
        pot_parents = self._graph[node_middle][Mark.Directed]
        if (node_x in pot_parents) and (node_y in pot_parents):
            return True
        else:
            return False

    def is_possible_collider(self, node_x, node_middle, node_y):
        """
        Test if node_2 is a possible collider in a completed version of this PAG.
        The given triplet should be either a triangle or the middle node should be a (definite) collider.
        This method is used, for example, to test if node_1 *--* node_2 *--* node_3 is a sub-path of a PDS-path.

        :param node_x: first node
        :param node_middle: middle (second) node
        :param node_y: third node
        :return: 'True' if the middle node is a possible collider
        """
        if node_x == node_y or \
                (not self.is_connected(node_x, node_middle)) or \
                (not self.is_connected(node_middle, node_y)):  # make sure there is a path: X *--* Middle *--* Y
            return False
        return self.is_connected(node_x, node_y) or \
               self.is_collider(node_middle=node_middle, node_x=node_x, node_y=node_y)

    def is_possible_parent(self, potential_parent_node, child_node):
        """
        Test if a node can possibly serve as parent of the given node.
        Make sure that on the connecting edge
            (a) there is no head edge-mark (->) at the tested node and
            (b) there is no tail edge-mark (--) at the given node,
        where variant edge-marks (o) are allowed.
        :param potential_parent_node: the node that is being tested
        :param child_node: the node that serves as the child
        :return:
        """
        if potential_parent_node == child_node:
            return False
        if not self.is_connected(potential_parent_node, child_node):
            return False

        if ((potential_parent_node in self._graph[child_node][Mark.Tail]) or
                (child_node in self._graph[potential_parent_node][Mark.Directed])):
            return False
        else:
            return True

        # if ((potential_parent_node in self._graph[child_node][Mark.Directed] or
        #      potential_parent_node in self._graph[child_node][Mark.Circle]) and
        #     (child_node in self._graph[potential_parent_node][Mark.Tail] or
        #      child_node in self._graph[potential_parent_node][Mark.Circle])):
        #     return True
        # else:
        #     return False

    def find_possible_children(self, parent_node, en_nodes=None):
        if en_nodes is None:
            en_nodes = self.nodes_set - {parent_node}
        potential_child_nodes = set()
        for potential_node in en_nodes:
            if self.is_possible_parent(potential_parent_node=parent_node, child_node=potential_node):
                potential_child_nodes.add(potential_node)

        return potential_child_nodes

    def find_possible_parents(self, child_node, en_nodes=None):
        if en_nodes is None:
            en_nodes = self.nodes_set - {child_node}

        possible_parents = {parent_node for parent_node in en_nodes if self.is_possible_parent(parent_node, child_node)}
        return possible_parents

    def is_parent(self, node_parent, node_child):
        """
        Test if a node is a parent of another parent: that is, there is a directed edge: node_source ---> node_target
        :param node_parent:
        :param node_child:
        :return: True if the relation exists in the graph; otherwise, False
        """
        return ((node_parent in self._graph[node_child][Mark.Directed]) and
                (node_child in self._graph[node_parent][Mark.Tail]))

    def find_parents(self, node_child):
        """
        Find the set of parents, oriented edges parent ---> child
        :param node_child:
        :return:
        """
        parent_nodes = set()
        potential_parents = self._graph[node_child][Mark.Directed]
        for potential_node in potential_parents:
            if node_child in self._graph[potential_node][Mark.Tail]:  # should have a tail at the parent
                parent_nodes.add(potential_node)
        return parent_nodes

    

    def find_uncovered_path(self, node_x, node_y, neighbor_x, neighbor_y, en_nodes=None, edge_condition=None):
        """
        Find a path <X, Neighbor_X, ..., Neighbor_Y, Y> such that for every three consecutive nodes <V, U, W>,
        V and W are disconnected.
        In general, the shortest path, excluding the end-points, has length 2 (a total of 4 nodes path);
        however, this function treats a special case when neighbor_x == neighbor_y by immediately (!) returning the path
        [neighbor_X] without (!) testing if node_x and node_y are disconnected, or testing edge_condition.
        :param node_x: one end of the path
        :param node_y: second end of the path
        :param neighbor_x: a node that must be considered as the start of the path: <A, neighbor_a, ..., B>.
        :param neighbor_y: a node that must be considered as the end of the path: <A, ..., neighbor_b, B>.
        :param en_nodes: nodes from which to construct the path
        :param edge_condition: a condition that restricts the relation between two consecutive nodes on the path.
            Note that it does not test the relation between X & Y and their given neighbors, neighbor_x & neighbor_y.
            Example 1: an uncovered circle path:
                edge_condition = lambda in1, in2: pag.is_edge(in1, in2, Mark.Circle, Mark.Circle)
            Example 2: a possible (potentially) directed uncovdered path:
                edge_condition = lambda in1, in2: pag.is_possible_parent(in1, in2)
        :return: path nodes excluding the end-points
        """
        # ToDo: needs to be thoroughly debuged
        if neighbor_x == neighbor_y:
            return [neighbor_x]

        if edge_condition is None:
            edge_condition = self.is_connected

        if en_nodes is None:
            en_nodes = self.nodes_set
        en_nodes = en_nodes - {node_x, node_y, neighbor_x, neighbor_y}

        # Exit condition of the recursion: a trivial uncovered path
        if edge_condition(neighbor_x, neighbor_y) and \
                (not self.is_connected(neighbor_x, node_y)) and \
                (not self.is_connected(node_x, neighbor_y)):
            return [neighbor_x, neighbor_y]  # found a trivial path

        # Find path extensions: node_x --- neighbor_x --- node_c
        # s.t. neighbor_x --- node_c is a qualifying edge and node_x is disconnected from node_c
        c_nodes = {tested_node for tested_node in self.find_adjacent_nodes(neighbor_x, en_nodes)
                   if edge_condition(neighbor_x, tested_node) and not self.is_connected(node_x, tested_node)}

        for node_c in c_nodes:
            path = self.find_uncovered_path(node_x=neighbor_x, node_y=node_y,
                                            neighbor_x=node_c, neighbor_y=neighbor_y,
                                            en_nodes=en_nodes, edge_condition=edge_condition)
            if path is not None:
                return [neighbor_x, *path]
        else:
            return None

    def find_discriminating_path_to_triplet(self, node_a, node_b, node_c, nodes_set=None):
        """
        Find a discriminating path from some node (denoted D) to node C for node B.
        That is, D *--> ? <--> ... <--> A <--* B *--* C
        :param node_a:
        :param node_b:
        :param node_c:
        :param nodes_set:
        :return: Path source node (node D)
        """
        if nodes_set is None:
            nodes_set = self.nodes_set - {node_a, node_b, node_c}  # create a copy

        # assumed: A <--* B *--* C or a path from A to B with all colliders & parents of C; and A ---> C,
        # we need to find D such that D *--> A, D and C are disjoint
        d_nodes = (self._graph[node_a][Mark.Directed] - {node_a, node_b, node_c}) & nodes_set
        new_a_nodes = set()
        for node_d in d_nodes:
            if not self.is_connected(node_d, node_c):  # found a discriminating path from D to C for B, (D, A, B, C)
                return node_d  # the source of the path
            else:
                # if D ---> C and D <--> A, then D becomes the "A" node in the new search triplet
                if self.is_parent(node_d, node_c) and node_a in self._graph[node_d][Mark.Directed]:
                    new_a_nodes.add(node_d)

        # didn't find a minimal discriminating path (containing three edges). Search with the new "A" nodes
        # we have D nodes that are part of the path D *--> A <--* B *--* C and D ---> C and A ---> C
        for new_node_a in new_a_nodes:
            node_d = self.find_discriminating_path_to_triplet(new_node_a, node_b, node_c, nodes_set - {new_node_a})
            if node_d is not None:
                return node_d

        return None  # didn't find a discriminating path

    # -----------------------------------------------------------------------------------------------------------------
    # --- Methods that modify the graph -------------------------------------------------------------------------------
    def copy_skeleton_from_pdag(self, pdag: PDAG, nodes_set=None):
        """
        Add the skeleton of in an external pdag (ignore edge-marks) to the current PAG (add o--o edges)
        :param pdag: An external PDAG object
        :param nodes_set: a set of nodes that define the graph to copy
        :return:
        """
        assert isinstance(pdag, PDAG)

        if nodes_set is None:
            nodes_set = self.nodes_set

        for node_i, node_j in combinations(nodes_set, 2):
            if pdag.is_connected(node_i, node_j):
                self.add_edge(node_i, node_j, Mark.Circle, Mark.Circle)

    def orient_v_structures(self, sepsets=None):
        """
        Orient X *--* Z *--* Y as X *--> Z <--* if X and Y are disjoint and Z is not in their separation set
        :param sepsets: Separating sets, an instance of the SeparationSet class
        :return:
        """
        assert sepsets is not None
        # check each node if it can serve as a collider for a disjoint neighbors
        for node_z in self.nodes_set:
            # check neighbors
            xy_nodes = self.find_adjacent_nodes(node_z)  # neighbors with some edge-mark at node_z
            for node_x, node_y in combinations(xy_nodes, 2):
                if self.is_connected(node_x, node_y):
                    continue  # skip this pair as they are connected
                if node_z not in sepsets.get_sepset(node_x, node_y):
                    self.replace_edge_mark(
                        node_source=node_x, node_target=node_z, requested_edge_mark=Mark.Directed)  # orient X *--> Z
                    self.replace_edge_mark(
                        node_source=node_y, node_target=node_z, requested_edge_mark=Mark.Directed)  # orient Y *--> Z

    def maximally_orient_pattern(self, rules_set=None):
        """
        Complete orienting graph. It is assumed that all v-structures have been previously oriented
        :param rules_set:
        :return:
        """
        if rules_set is None:
            rules_set = list(self.orientation_rules.keys())

        graph_modified = True
        while graph_modified:
            graph_modified = False
            for rule_idx in rules_set:
                rule = self.orientation_rules[rule_idx]
                graph_modified |= rule()

    # Batch of rules for initial graph orientation [R1, R2, R3, R4] ----------------------------------------------------
    def orient_by_rule_1(self):
        """
        [R1] If A *--> B o--* C, and A & C are not connected, then orient A *--> B ---> C
        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            a_nodes = self._graph[node_b][Mark.Directed]
            c_nodes = self._graph[node_b][Mark.Circle].copy()
            for node_a in a_nodes:
                for node_c in c_nodes:
                    if not self.is_connected(node_a, node_c):
                        self.replace_edge_mark(node_source=node_c, node_target=node_b,
                                               requested_edge_mark=Mark.Tail)  # tail edge-mark
                        self.replace_edge_mark(node_source=node_b, node_target=node_c,
                                               requested_edge_mark=Mark.Directed)  # head edge-mark
                        graph_modified = True

        return graph_modified

    def orient_by_rule_2(self):
        """
        [R2] If (1) A *--> B ---> C or (2) A ---> B *--> C, and A *--o C, then orient A *--> C

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False

        # case (1): If A *--> B ---> C and A *--o C, then orient A *--> C
        for node_b in self.nodes_set:
            a_nodes = self._graph[node_b][Mark.Directed]  # A *--> B
            c_nodes = self._graph[node_b][Mark.Tail]  # B ---* C (we still need to check that B ---> C)

            for node_c in c_nodes:
                if node_b not in self._graph[node_c][Mark.Directed]:  # check if B *--> C (already is B ---* C)
                    continue  # skip this node_c
                # now we are sure that B ---> C
                for node_a in a_nodes:
                    if node_a in self._graph[node_c][Mark.Circle]:  # if A *--o C
                        self.replace_edge_mark(node_source=node_a, node_target=node_c,
                                               requested_edge_mark=Mark.Directed)
                        graph_modified = True

        # case (2): If A ---> B *--> C, and A *--o C, then orient A *--> C
        for node_c in self.nodes_set:
            b_nodes = self._graph[node_c][Mark.Directed].copy()  # B *--> C

            for node_b in b_nodes:
                a_nodes = self._graph[node_b][Mark.Directed]  # A *--> B (we still need to check A ---> B)
                for node_a in a_nodes:
                    if node_b not in self._graph[node_a][Mark.Tail]:  # check if A ---* B (already is A *--> B)
                        continue  # skip this node_x
                    if node_a in self._graph[node_c][Mark.Circle]:  # if A *--o C
                        self.replace_edge_mark(node_source=node_a, node_target=node_c,
                                               requested_edge_mark=Mark.Directed)
                        graph_modified = True

        return graph_modified

    def orient_by_rule_3(self):
        """
        [R3] If A *--> B <--* C and A *--o D o--* C, A & C not connected, D *--o B, then orient D *--> B

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            d_nodes = self._graph[node_b][Mark.Circle].copy()  # D *--o B
            for node_d in d_nodes:
                # find pairs that satisfy (A, C) *--> B and (A, C) *--o D
                ac_nodes = self._graph[node_b][Mark.Directed] & self._graph[node_d][Mark.Circle]
                for (node_a, node_c) in combinations(ac_nodes, 2):
                    if not self.is_connected(node_a, node_c):  # a pair (A,C) exists and is disjoint
                        self.replace_edge_mark(node_source=node_d, node_target=node_b,
                                               requested_edge_mark=Mark.Directed)
                        graph_modified = True

        return graph_modified

    def orient_by_rule_4(self):
        """
        [R4] If a discriminating path between D and C for B, i.e., (D, ..., A, B, C), and B o--* C, then:
            (1) if B in sep-set of (D, C), orient: B ---> C,
            (2) else, orient the triplet A <--> B <--> C

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            c_nodes = self._graph[node_b][Mark.Circle].copy()  # B o--* C
            for node_c in c_nodes:
                potential_a_nodes = self.find_parents(node_c)  # should comply with A ---> C
                for node_a in potential_a_nodes:
                    if node_b in self._graph[node_a][Mark.Directed]:  # should comply with A <--* B
                        # node_x is legal
                        node_d = self.find_discriminating_path_to_triplet(node_a, node_b, node_c)
                        if node_d is not None:
                            # found a discriminating path
                            if node_b in self.sepset.get_sepset(node_d, node_c):
                                # orient B o--* C into B ---> C
                                self.replace_edge_mark(node_source=node_b, node_target=node_c,
                                                       requested_edge_mark=Mark.Directed)
                                self.replace_edge_mark(node_source=node_c, node_target=node_b,
                                                       requested_edge_mark=Mark.Tail)
                            else:
                                # orient A <--> B <--> C
                                self.replace_edge_mark(node_source=node_b, node_target=node_a,
                                                       requested_edge_mark=Mark.Directed)
                                self.replace_edge_mark(node_source=node_a, node_target=node_b,
                                                       requested_edge_mark=Mark.Directed)
                                self.replace_edge_mark(node_source=node_b, node_target=node_c,
                                                       requested_edge_mark=Mark.Directed)
                                self.replace_edge_mark(node_source=node_c, node_target=node_b,
                                                       requested_edge_mark=Mark.Directed)

                            graph_modified = True

        return graph_modified

    # Batch of rules [R5, R6, R7] when considering selection bias ------------------------------------------------------
    def orient_by_rule_5(self):
        """
        [R5] If A o--o B and there is an uncovered circle path <A, X, ..., Y, B>,
        such that (A, Y) are disconnected and (X, B) are disconnected

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False

        # create a list of all the A o--o B edges in the graph
        var_edges_list = [(node_a, node_b)
                          for node_a in self.nodes_set
                          for node_b in self.nodes_set
                          if self.is_edge(node_a, node_b, Mark.Circle, Mark.Circle)]

        # examine each variant edge
        for (node_a, node_b) in var_edges_list:
            a_neighbors_list = {nb_a for nb_a in self.nodes_set - {node_a, node_b}
                                if self.is_edge(node_a, nb_a, Mark.Circle, Mark.Circle)  # node_x o--o nb_a
                                and not self.is_connected(node_b, nb_a)}  # nb_a not connected to node_y
            b_neighbors_list = {nb_b for nb_b in self.nodes_set - {node_a, node_b}
                                if self.is_edge(node_b, nb_b, Mark.Circle, Mark.Circle)  # node_y o--o nb_b
                                and not self.is_connected(node_a, nb_b)}  # nb_b not connected to node_x

            for neighbor_a in a_neighbors_list:
                for neighbor_b in b_neighbors_list:
                    uncov_circ_path = \
                        self.find_uncovered_path(node_a, node_b,
                                                 neighbor_x=neighbor_a, neighbor_y=neighbor_b, edge_condition=
                                                 lambda in1, in2: self.is_edge(in1, in2, Mark.Circle, Mark.Circle))
                    if uncov_circ_path is not None:
                        # criterion is met
                        graph_modified = True
                        self.reset_orientations(Mark.Tail, {node_a, node_b})
                        full_path = [node_a, *uncov_circ_path, node_b]  # add the end-points, A and B, to the path
                        for idx in range(len(full_path)-1):
                            self.reset_orientations(Mark.Tail, {full_path[idx], full_path[idx+1]})

        return graph_modified

    def orient_by_rule_6(self):
        """
        [R6] If A ---- B o--* C, and A & C may or may not be connected, then orient B ---* C

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            a_nodes = {can_node for can_node in self._graph[node_b][Mark.Tail]
                       if node_b in self._graph[can_node][Mark.Tail]}  # A ---- B
            c_nodes = self._graph[node_b][Mark.Circle].copy()  # B o--* C
            for node_a in a_nodes:
                for node_c in c_nodes:
                    self.replace_edge_mark(node_source=node_c, node_target=node_b,
                                           requested_edge_mark=Mark.Tail)  # tail edge-mark
                    graph_modified = True

        return graph_modified

    def orient_by_rule_7(self):
        """
        [R7] If A ---o B o--* C, and A & C are not connected, then orient tail B ---* C

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            a_nodes = {can_node for can_node in self._graph[node_b][Mark.Circle]
                       if node_b in self._graph[can_node][Mark.Tail]}  # A ---o B
            c_nodes = self._graph[node_b][Mark.Circle].copy()  # B o--* C
            for node_a in a_nodes:
                for node_c in c_nodes:
                    if node_a == node_c:
                        continue
                    if not self.is_connected(node_a, node_c):
                        self.replace_edge_mark(node_source=node_c, node_target=node_b,
                                               requested_edge_mark=Mark.Tail)  # tail edge-mark
                        graph_modified = True

        return graph_modified

    # Batch of rules [R8, R9, R10] -------------------------------------------------------------------------------------
    def orient_by_rule_8(self):
        """
        [R8] If A ---> B ---> C or A ---o B ---> C and A o--> C then orient tail A ---> C

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            a_nodes = {can_node for can_node in self._graph[node_b][Mark.Directed] | self._graph[node_b][Mark.Circle]
                       if node_b in self._graph[can_node][Mark.Tail]}  # A ---> B or A ---o B
            c_nodes = {can_node for can_node in self._graph[node_b][Mark.Tail]
                       if node_b in self._graph[can_node][Mark.Directed]}  # B ---> C
            for node_a in a_nodes:
                for node_c in c_nodes:
                    if self.is_edge(node_a, node_c, Mark.Circle, Mark.Directed):  # A o--> C
                        self.replace_edge_mark(node_source=node_c, node_target=node_a,
                                               requested_edge_mark=Mark.Tail)  # tail edge-mark at A
                        graph_modified = True

        return graph_modified

    def orient_by_rule_9(self):
        """
        [R9] If A o--> C and there is a possibly directed uncovered path <A, B, ..., D, C>, B and C are not connected
                then orient tail A ---> C.

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_a in self.nodes_set:
            c_nodes = {can_node for can_node in self._graph[node_a][Mark.Circle]
                       if node_a in self._graph[can_node][Mark.Directed]}  # A o--> C
            for node_c in c_nodes:
                # look for a possibly directed uncovered path s.t. B and C are not connected (for the given A o--> C
                b_nodes = {can_b for can_b in self.find_possible_children(node_a, self.nodes_set - {node_c, node_a})
                           if not self.is_connected(can_b, node_c)}

                for node_b in b_nodes:
                    d_nodes = self.find_possible_parents(node_c, self.nodes_set - {node_a, node_b, node_c})
                    # search a p.d. uncovered path for <A, B, ..., C>
                    for node_d in d_nodes:
                        pd_path = self.find_uncovered_path(node_x=node_a, node_y=node_c,
                                                           neighbor_x=node_b, neighbor_y=node_d,
                                                           edge_condition=self.is_possible_parent)
                        if pd_path is not None:
                            self.replace_edge_mark(node_source=node_c, node_target=node_a,
                                                   requested_edge_mark=Mark.Tail)  # tail edge-mark at A
                            graph_modified = True
                            return graph_modified

        return graph_modified

    def orient_by_rule_10(self):
        """
        [R10] If A o--> C and B ---> C <---D, and if there are two possibly directed uncovered paths
        <A, E, ..., B>, <A, F, ..., D> s.t. E, F are disconnected, and any of these paths can be a single-edge path,
        A o--> B or A o--> D, then orient tail A ---> C.

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False

        for node_c in self.nodes_set:
            a_nodes = {can_node for can_node in self.nodes_set - {node_c}
                       if self.is_edge(can_node, node_c, Mark.Circle, Mark.Directed)}  # find A o--> C
            if len(a_nodes) == 0:
                continue  # no A o--> was found for the specific C node, go to the next c_node
            # find B, D such that B ---> C <---D (directed edges)
            bd_nodes = {can_node for can_node in self.nodes_set - {node_c}
                        if self.is_edge(can_node, node_c, Mark.Tail, Mark.Directed)}  # find a pair {D,B} ---> C
            if len(bd_nodes) < 2:
                continue
            for node_a in a_nodes:  # try to orient the tail of this specific A o--> C edge
                a_possible_children = self.find_possible_children(
                    parent_node=node_a, en_nodes=self.nodes_set - {node_a, node_c})  # find A o--{o,>} neighbors
                if len(a_possible_children) < 2:
                    continue  # cannot draw two paths out of A so go to the next A-node

                # now all the nodes are specified test the condition of the rule
                for node_b, node_d in combinations(bd_nodes, 2):
                    # try to construct two p.d. uncovered paths
                    # note that a path <A, E, ...> may end in either B or D. The same for <A, F, ...>
                    for node_e in a_possible_children:
                        for node_f in a_possible_children:
                            if node_e == node_f or self.is_connected(node_e, node_f):
                                continue

                            path_e = self.find_uncovered_path(node_x=node_a, node_y=node_c,
                                                              neighbor_x=node_e, neighbor_y=node_b)
                            if path_e is not None:
                                path_f = self.find_uncovered_path(node_x=node_a, node_y=node_c,
                                                                  neighbor_x=node_f, neighbor_y=node_d)
                                if path_f is not None:
                                    self.replace_edge_mark(node_source=node_c, node_target=node_a,
                                                           requested_edge_mark=Mark.Tail)  # tail edge-mark at A
                                    graph_modified = True
                                    return graph_modified

        return graph_modified
