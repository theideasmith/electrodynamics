import numpy as np


def avg_distance(adjacency, node_A, node_B, directed=False, weighted=False):
    """
    Finds the average distance between node A and B

    =============   ======================================
    Varname         Description
    =============   ======================================
    adjacency       Adjacency is an n*n adjacency matrix

    node_A          0<=node_A<=n
    node_B          0<=node_B<=n
        note: if node_A = node_B: return 0

    directed        Whether the graph is directed. If not
                    we assume graph is symmetric along the
                    diagnal and only use the upper triangle
                    of the graph for computations

    weighted        If True, the weights of the edges will be
                    summed to compute net length. If not,
                    returns the avg number of edges between
                    node_A and node_B

    """
