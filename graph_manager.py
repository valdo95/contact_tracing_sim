import networkx as nx
import matplotlib.pyplot as ml
import random


def create_school_graph(nodes_list, density=0.05):
    graph = nx.erdos_renyi_graph(len(nodes_list), density)
    mapping = dict(zip(list(graph), nodes_list))
    print(nodes_list)
    print(mapping)
    return nx.relabel_nodes(graph, mapping, copy=False)


def create_work_graph(nodes_list, density=0.30):
    graph = nx.erdos_renyi_graph(len(nodes_list), density)
    mapping = dict(zip(list(graph), nodes_list))
    print(nodes_list)
    print(mapping)
    return nx.relabel_nodes(graph, mapping, copy=False)


def create_home_graph(nodes_list):
    graph = nx.complete_graph(len(nodes_list))
    mapping = dict(zip(list(graph), nodes_list))
    res = nx.relabel_nodes(graph, mapping, copy=True)
    type_graph = []
    nx.set_node_attributes(res, type_graph, "graph_name")
    type_graph.append("home")
    return res


def create_station_graph(nodes_list, density=0.1):
    graph = nx.erdos_renyi_graph(len(nodes_list), density)  # watts_strogatz_graph(len(nodes_list), 3, density)
    mapping = dict(zip(list(graph), nodes_list))
    res = nx.relabel_nodes(graph, mapping, copy=True)
    type_graph = []
    nx.set_node_attributes(res, type_graph, "graph_name")
    type_graph.append("station")
    return res


def create_tube_graph(nodes_list, density=0.2):
    graph = nx.erdos_renyi_graph(len(nodes_list), density)  # watts_strogatz_graph(len(nodes_list), 3, density)
    mapping = dict(zip(list(graph), nodes_list))
    res = nx.relabel_nodes(graph, mapping, copy=True)
    type_graph = []
    nx.set_node_attributes(res, type_graph, "graph_name")
    type_graph.append("tube")
    return res


def read_graph(name):
    graph = nx.read_adjlist(str(name) + ".adjlist")
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default', label_attribute=None)
    return graph


def write_graph(graph, name):
    nx.write_adjlist(graph, name + ".adjlist")


def write_labeled_graph(graph, name):
    # nx.write_graphml(graph, name + ".graphml")
    nx.write_gml(graph, "test.gml")


def read_labeled_graph(name):
    return nx.read_gml(name + ".gml")


def print_graph(graph, name):
    print(name + " graph")
    print("number of nodes ............... " + str(nx.number_of_nodes(graph)))
    print("number of edges ............... " + str(nx.number_of_edges(graph)))
    print("density ....................... " + str(nx.density(graph)))
    print("graph edges: ")
    print(list(nx.edges(graph)))
    print("graph nodes: ")
    print(sorted(list(graph)))
    nx.draw_circular(graph)
    # nx.draw_networkx_edge_labels(graph)
    ml.savefig(name + "graph.png")
    ml.close()

# def create_home_graph(n):
#     graph = nx.Graph()
#     for i in range(0, n):
#         graph.add_node(i)
#     for elem in graph:
#         for nb in graph:
#             graph.add_edge(elem, nb)
#     print("home graph")
#     return graph


# def create_school_graph_old(n):
#     density = 0.10
#     graph = nx.erdos_renyi_graph(n, density)
#     return graph
#
#
# def create_work_graph(n):
#     density = 0.10
#     graph = nx.erdos_renyi_graph(n, density)
#     return graph
#
#
# def create_station_graph(n):
#     density = 0.10
#     graph = nx.erdos_renyi_graph(n, density)
#     for (u, v, w) in graph.edges(data=True):
#         w['weight'] = random.randint(0, 10)
#
#     return graph
