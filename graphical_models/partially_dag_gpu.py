from itertools import combinations
from .basic_equivalance_class_graph_gpu import MixedGraph
from . import arrow_head_types as Mark


class gpuPDAG(MixedGraph):
    """
    Partially directed graph having two type of arrowheads: directed (--> node) and undirected (--- node)
    """
    def __init__(self, nodes_set):
        super().__init__(nodes_set, [Mark.Undirected, Mark.Directed])
        self.orientation_rules = {
            1: self.orient_by_rule_1,
            2: self.orient_by_rule_2,
            3: self.orient_by_rule_3,
            4: self.orient_by_rule_4
        }

    # --- graph initialization functions ------------------------------------------------------------------------------
    def create_complete_graph(self, nodes_set=None):
        super().create_complete_graph(Mark.Undirected, nodes_set)

    # --- graph query functions ---------------------------------------------------------------------------------------
    def parents(self, target_node):
        """
        Return the (directed) parents of the target node
        :param target_node: the child node
        :return: parents of target_node
        """
        return self._graph[target_node][Mark.Directed]

    

    def undirected_neighbors(self, node):
        """
        Return neighbors connected by an un-directed edge
        :param node: a given node
        :return: neighbors of node connected by an un-directed edge
        """
        return self._graph[node][Mark.Undirected]

    
    def fan_in(self, target_node):
        """
        Return the number of arrow heads (directed and undirected) into a node
        :param target_node: a node
        :return: Fan-in of node target_node
        """
        return len(self.parents(target_node)) + len(self.undirected_neighbors(target_node))

   

    # --- functions that modify the graph -----------------------------------------------------------------------------
    def orient_edge(self, source_node, target_node):
        """
        Modify the graph by orienting an undirected edge source --- target to source --> target
        Note that the existence of an undirected edge is not tested in order to allow
        bi-directed edges (spurious association)
        :param source_node: to be a parent node
        :param target_node: to be a child node
        :return:
        """
        self._graph[target_node][Mark.Directed].add(source_node)  # add a directed arrow head
        self._graph[target_node][Mark.Undirected].discard(source_node)  # remove an undirected arrow head
        self._graph[source_node][Mark.Undirected].discard(target_node)  # remove an undirected arrow head

    def delete_directed_edge(self, source_node, target_node):
        """
        Delete a directed edge
        :param source_node:
        :param target_node:
        :return:
        """
        self._graph[target_node][Mark.Directed].discard(source_node)

    def delete_undirected_edge(self, node_i, node_j):
        """
        Delete an undirected edge
        :param node_i: 1st node
        :param node_j: 2nd node
        :return:
        """
        self._graph[node_i][Mark.Undirected].discard(node_j)
        self._graph[node_j][Mark.Undirected].discard(node_i)

    def delete_edge(self, node_i, node_j):
        self.delete_directed_edge(node_i, node_j)  # delete directed arrow head into node j
        self.delete_directed_edge(node_j, node_i)  # delete directed arrow head into node i
        self.delete_undirected_edge(node_i, node_j)  # delete undirected arrow heads

   

    def orient_by_rule_1(self, en_nodes):
        """
        [R1] Orient Z --> X --- Y into Z --> X --> Y if Z and Y are not connected.
        Orient undirected edges connected to endogenous nodes.
        :param en_nodes: endogenous nodes to be tested
        :return: True if an orientation is found, False if graph is unchanged
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()  # neighbors of the current Y
            for node_x in x_nodes:  # test all undirected edges "into" Y
                for node_z in self.parents(node_x):
                    if not self.is_connected(node_z, node_y):
                        self.orient_edge(source_node=node_x, target_node=node_y)  # orient X --> Y
                        graph_modified = True
                        break  # X --> Y was oriented so stop searching through Z nodes and go to the next X --- Y

        return graph_modified

    def orient_by_rule_2(self, en_nodes):
        """
        [R2] Orient X --- Y into X --> Y if there is a directed path X --> Z --> Y (utilizing acyclic assumption).
        Orient undirected edges connected to endogenous nodes.
        :param en_nodes: endogenous nodes concicting the sub-graph to be oriented
        :return: True if an orientation is found, False if graph is unchanged
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            for node_x in x_nodes:
                z_nodes = self.parents(node_y)  # directed parents
                for node_z in z_nodes:
                    if node_x in self.parents(node_z):
                        self.orient_edge(source_node=node_x, target_node=node_y)
                        graph_modified = True
                        break  # X --> Y was oriented so stop searching through Z nodes and go to the next X --- Y

        return graph_modified

    def orient_by_rule_3(self, en_nodes):
        """
        [R3] Orient X --- Y into X --> Y if there exists X --- W --> Y and X --- Z --> Y, where W and Z are disconnected
        Orient undirected edges connected to endogenous nodes.
        :param en_nodes: endogenous nodes
        :return: True if an orientation is found, False if graph is unchanged
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            wz_nodes = self.parents(node_y)
            for node_x in x_nodes:
                wz_nodes_of_x = self.undirected_neighbors(node_x).intersection(wz_nodes)  # W,Z neighbors of X
                for node_w, node_z in combinations(wz_nodes_of_x, 2):
                    if self.is_connected(node_w, node_z):
                        continue  # skip as W and Z are connected

                    self.orient_edge(source_node=node_x, target_node=node_y)
                    graph_modified = True
                    break  # X --> Y was oriented so stop searching through Z nodes and go to the next X --- Y

        return graph_modified

    def orient_by_rule_4(self, en_nodes):
        """
        [R4] Orient X --- Y into X --> Y if W --> Z --> Y and X and Z are connected by an undirected edge,
        and W and Y are disconnected.
        Orient undirected edges connected to endogenous nodes.
        :param en_nodes: endogenous nodes
        :return: True if an orientation is found, False if graph is unchanged
        """
        graph_updated = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            z_nodes = self.parents(node_y)
            for node_x in x_nodes:
                for node_z in z_nodes:
                    if not self.is_connected(node_z, node_x):  # make sure Z and X are connected
                        continue  # skip and search for the next Z for the given X node
                    w_nodes = self.parents(node_z).intersection(self.undirected_neighbors(node_x))
                    if len(w_nodes) > 0:
                        self.orient_edge(source_node=node_x, target_node=node_y)
                        graph_updated = True
                        break

        return graph_updated

    def copy(self):
        """
        Copy graph
        :return: a PDAG copy
        """
        target_pdag = PDAG(self.nodes_set)

        for node in self.nodes_set:
            target_pdag._graph[node][Mark.Undirected] = self._graph[node][Mark.Undirected].copy()
            target_pdag._graph[node][Mark.Directed] = self._graph[node][Mark.Directed].copy()

        return target_pdag