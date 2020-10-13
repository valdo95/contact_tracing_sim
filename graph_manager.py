import networkx as nx
import matplotlib.pyplot as ml
import random


def create_home_graph(n):
    graph = nx.Graph()
    for i in range(0, n):
        graph.add_node(i)
    for elem in graph:
        for nb in graph:
            graph.add_edge(elem, nb)
    print("home graph")
    return graph


def create_home_graph(nodes_list):
    graph = nx.complete_graph(len(nodes_list))
    mapping = dict(zip(list(graph), nodes_list))
    # print(nodes_list)
    # print(mapping)
    return nx.relabel_nodes(graph, mapping, copy=False)


def create_school_graph(n):
    density = 0.10
    graph = nx.erdos_renyi_graph(n, density)
    return graph


def create_work_graph(n):
    density = 0.10
    graph = nx.erdos_renyi_graph(n, density)
    return graph


def create_station_graph(n):
    density = 0.10
    graph = nx.erdos_renyi_graph(n, density)
    for (u, v, w) in graph.edges(data=True):
        w['weight'] = random.randint(0, 10)

    return graph


def read_graph(name):
    return nx.read_adjlist(name + ".adjlist")


def write_graph(graph, name):
    nx.write_adjlist(graph, name + ".adjlist")


def print_graph(graph, name):
    print(name + " graph")
    print("number of nodes ............... " + str(nx.number_of_nodes(graph)))
    print("number of edges ............... " + str(nx.number_of_edges(graph)))
    print("density ....................... " + str(nx.density(graph)))
    nx.draw_circular(graph)
    # nx.draw_networkx_edge_labels(graph)
    ml.savefig(name + "graph.png")
    ml.close()
