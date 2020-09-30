import graph_manager as gm


if __name__ == '__main__':
    graph = gm.create_home_graph(6)
    g2 = gm.create_school_graph(20)
    gm.print_graph(g2)


