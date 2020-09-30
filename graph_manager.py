import networkx as nx
import matplotlib.pyplot as ml


def create_home_graph(n):
    graph = nx.Graph()
    for i in range(0, n):
        graph.add_node(i)
    for elem in graph:
        for nb in graph:
            graph.add_edge(elem, nb)
    print("home graph")
    return graph


def create_school_graph(n):
    density = 0.30
    graph = nx.erdos_renyi_graph(n, density)
    return graph


def create_work_graph(n):
    density = 0.10
    graph = nx.erdos_renyi_graph(n, density)
    return graph


def create_station_graph(n):
    density = 0.10
    graph = nx.erdos_renyi_graph(n, density)
    return graph


def print_graph(graph):
    print("number of nodes ............... " + str(nx.number_of_nodes(graph)))
    print("number of edges ............... " + str(nx.number_of_edges(graph)))
    print("density ....................... " + str(nx.density(graph)))
    nx.draw_circular(graph)
    # nx.draw_networkx_edge_labels(graph)
    ml.savefig("graph.png")
