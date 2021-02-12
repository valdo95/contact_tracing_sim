import networkx as nx
import matplotlib.pyplot as ml
import random
import time


def remapping_nodes(graph, nodes_list):
    mapping = dict(zip(list(graph), nodes_list))
    res = nx.relabel_nodes(graph, mapping, copy=True)  # make a copy is more efficient, but You have to clear old graph
    graph.clear()
    return res


def set_graph_name(graph, name):
    type_graph = []
    nx.set_node_attributes(graph, type_graph, "graph_name")
    type_graph.append(name)


def create_school_graph(nodes_list, density=0.4, rg=None):
    graph_name = "school"
    graph = nx.erdos_renyi_graph(len(nodes_list), density, seed=rg)
    res = remapping_nodes(graph, sorted(nodes_list))
    graph.name = "School"
    #set_graph_name(res, graph_name)
    return res


def create_office_graph(nodes_list, density=0.3, rg=None):
    graph_name = "office"
    graph = nx.erdos_renyi_graph(len(nodes_list), density, seed=rg)  # watts_strogatz_graph(len(nodes_list), 3, density)
    res = remapping_nodes(graph, sorted(nodes_list))
    graph.name = "Office"
    #set_graph_name(res, graph_name)
    return res


def create_home_graph(nodes_list):
    graph_name = "home"
    graph = nx.complete_graph(len(nodes_list))
    res = remapping_nodes(graph, nodes_list)
    graph.name = "Home"
    #set_graph_name(res, graph_name)
    return res


def create_station_graph(nodes_list, density=0.05, rg=None):
    graph_name = "station"
    # print("Start generation graph... ")
    # start_time = time.time()
    graph = nx.erdos_renyi_graph(len(nodes_list), density, seed=rg)  # watts_strogatz_graph(len(nodes_list), 3, density)
    # end_time = time.time()
    # duration = round((end_time - start_time), 3)
    # print("duration erdos reni: " + str(duration) + " Seconds")
    # start_time = time.time()
    # print("Start relabeling... ")
    res = remapping_nodes(graph, nodes_list)
    graph.name = "Station"
    # end_time = time.time()
    # duration = round((end_time - start_time), 3)
    # print("duration relabeling: " + str(duration) + " Seconds")
    #set_graph_name(res, graph_name)
    return res


def create_public_transport_graph(nodes_list, density=0.2, rg=None):
    graph_name = "public_transport"
    graph = nx.erdos_renyi_graph(len(nodes_list), density, seed=rg)  # watts_strogatz_graph(len(nodes_list), 3, density)
    res = remapping_nodes(graph, sorted(nodes_list))
    graph.name = "Transport"
    #set_graph_name(res, graph_name)
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
    # nx.draw_netofficex_edge_labels(graph)
    ml.savefig(name + "graph.png")
    ml.close()


def print_graph_with_labels(graph):
    for elem in graph.nodes:
        print("Id: " + str(elem) + "  Graph Name: " + str(graph.nodes[elem]["graph_name"]))


def print_graph_with_labels_and_neighb(graph):
    for elem in graph.nodes:
        print("Id: " + str(elem) + "     Graph Name: " + str(
            "-") + "    Neighbs List: " + str(list(nx.neighbors(graph, elem))))
