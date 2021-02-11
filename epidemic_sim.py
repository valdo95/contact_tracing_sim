import graph_manager as gm
import random
import EoN as eon  # not used
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from itertools import islice
# from random import randint
import time
import yaml
import sys
import gc
import file_manager as fm

home_step = 0  # step for days
work_step = 0  # step for days
n_days = 0  # number of days
n_days_quar = 0  # quarantine length
n_days_isol = 0  # isolation length
beta = 0  # probability of contagion
sigma = 0  # transition rate from E to I
gamma = 0  # transition rate from I to R
fr_inf = 0  # number of initial infected
step_p_day = 0  # number of step per day
n_step_home = 0  # number of step at home
n_step_work = 0  # number of step at work/school
n_step_transp = 0  # number of step on public transport
fr_station_user = 0  # fraction of people that use train station
fr_symptomatic = 0  # fraction of symptomatic
initial_seed = 0
rg1 = None
fam_graph = None
office_school_graph = None
transp_graph = None
# COUNTS
n = 0  # number of total node
n_inf = 0  # number of initial infected
n_stud = 0  # number of student
n_employs = 0  # number of employees
n_app_users = 0  # number of people with the app

pr_diagnosis = 0  # Prob of been diagnosed
pr_notification = 0  # Prob of receive a Notification

# Partitons size
min_family_size = 0
max_familiy_size = 0
min_school_size = 0
max_school_size = 0
min_office_size = 0
max_office_size = 0
min_transp_size = 0
max_transp_size = 0

# s_list = []  # list of susceptible nodes
# e_list = []  # list of exposed nodes
# i_list = []  # list of infected nodes
# r_list = []  # list of recovered/isolated/dead nodes

seir_list = []  # 0-->S, 1-->E, 2-->I, 3-->R, 4-->Isol., 5-->Quarantine
sir_list = []  # 0-->S, 1-->I, 2-->R
res_time_list = []  # res. times (if the node is I or E)
s_t = []  # number of susceptible for each step (e. g. step_p_day = 10 -> step = 1 / 10)
e_t = []  # number of exposed for each step
i_t = []  # number of infected for each step
r_t = []  # number of recovered/isolated/dead for each step

people_tot = []  # array of nodes
app_people = []  # One entry for each person: The values are True if the person use the app
contact_list = []  # [ngb, timestamp] contact list
# people_status = []  # One entry for each person: 0 -> S, value > 0 --> value represent the residue time in quarantine o isolation

# commuter_partitions = []  # list of list of people that use one specific station
public_transport_users = []  # list of list of people that use one specific public_transport/bus


def set_random_stream():
    if initial_seed != 0:
        return random.Random(initial_seed)
    else:
        return random.Random()


def generate_partitions(input_list, min_size=1, max_size=6):
    it = iter(input_list)
    while True:
        nxt = list(islice(it, rg1.randint(min_size, max_size)))
        if nxt:
            yield nxt
        else:
            break


def create_partions():
    global people_tot
    global fr_station_user

    rg1.shuffle(people_tot)
    n_station_user = int(fr_station_user * n)
    station_partitions = list(generate_partitions(people_tot[:n_station_user], 20, 100))
    public_transport_partitions = list(generate_partitions(people_tot[n_station_user:], 5, 40))
    return [station_partitions, public_transport_partitions]


def create_families_network():
    rg1.shuffle(people_tot)
    families = list(generate_partitions(people_tot, min_family_size, max_familiy_size))
    # print("families: " + str(families))
    temp_graphs = []
    for family in families:
        temp_graphs.append(gm.create_home_graph(family))
    return gm.nx.union_all(temp_graphs)


def create_transport_network():
    rg1.shuffle(people_tot)
    transp_part = list(generate_partitions(people_tot, min_transp_size, max_transp_size))
    # print("families: " + str(families))
    temp_graphs = []
    for bus in transp_part:
        temp_graphs.append(gm.create_public_transport_graph(bus))
    return gm.nx.union_all(temp_graphs)


def create_school_work_network():
    rg1.shuffle(people_tot)
    office_partitions = list(generate_partitions(people_tot[:n_stud], min_office_size, max_office_size))
    school_partitions = list(
        generate_partitions(people_tot[n_stud:(n_stud + n_employs)], min_school_size, max_school_size))
    # print("office partitions: " + str(office_partitions))
    # print("school partitions: " + str(school_partitions))
    temp_graphs = []
    for office in office_partitions:
        temp_graphs.append(gm.create_office_graph(office, 0.3))
    for school in school_partitions:
        temp_graphs.append(gm.create_school_graph(school, 0.3))
    return gm.nx.union_all(temp_graphs)


def compare_with_EON(comparison_type):
    global n
    global n_days
    global beta
    global gamma
    global n_inf
    global step_p_day
    with open("config_files/" + str(comparison_type) + "_input.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        n = data["n_nodes"]  # number of total node
        n_days = data["n_days"]  # number of days
        beta = data["beta"]
        gamma = data["gamma"]
        fr_inf = data["fr_inf"]
        step_p_day = data["step_p_day"]
        print("\nGraph and Time Parameters: \n")
        print("Number of Nodes: .......... " + str(n))
        print("n days: ................... " + str(n_days))
        print("Step per Day............... " + str(step_p_day))

        print("\nEpidemic Parameters: \n")
        print("Beta: ..................... " + str(beta))
        print("Gamma: .................... " + str(gamma))
        print("Fract Infected: ........... " + str(fr_inf))
        print()

    # gamma = 1 - numpy.nextafter(1., 0.)
    # sigma = 50  # sys.maxsize #genero numero molto grande
    n_inf = int(fr_inf * n)
    initialize_sir()
    graph = gm.nx.erdos_renyi_graph(n, 0.2)
    input()
    print("Start EON Simulation")
    epidemic(graph, n_days)

    print("End EON Simulation")
    input()
    flush_structures()
    print_SIR_count()
    initialize_sir()
    # print(sir_list)
    # print(res_time_list)
    # input()
    print("Start sim_SIR Simulation")
    sim_SIR_eff(graph, 0, n_days * step_p_day)
    print("End sim_SIR Simulation")
    plot_SIR_result("comparison_sim")


# def validation_0():
#     global n_days
#     global clock
#     global n
#     global beta
#     global gamma
#     global sigma
#     global fr_inf
#     global s_list
#     global e_list
#     global i_list
#     global r_list
#     global step_p_day
#     global people_tot
#
#     n = 3000
#     n_days = 50
#     beta = 0.02
#     sigma = 0.2
#     gamma = 0.1
#     fr_inf = 0.02
#     step_p_day = 1
#     initialize()
#     graph = gm.nx.erdos_renyi_graph(n, 0.05)
#     initialize_Infected()
#     interval_tot = n_days * step_p_day
#     sim_SEIR(graph, 0, interval_tot)
#     count_SEIR()

#
# def first_validation_old(validation_type):
#     global n_days
#     global n
#     global beta
#     global gamma
#     global sigma
#     global n_inf
#     global s_list
#     global i_list
#     global r_list
#     global step_p_day
#     global people_tot
#     global initial_seed
#
#     with open("config_files/" + str(validation_type) + "_input.yaml", 'r') as stream:
#         data = yaml.safe_load(stream)
#         n = data["n_nodes"]  # number of total node
#         n_days = data["n_days"]  # number of days
#         n_days = data["n_days"]
#         beta = data["beta"]
#         sigma = data["sigma"]
#         gamma = data["gamma"]
#         fr_inf = data["fr_inf"]
#         initial_seed = data["seed"]
#         step_p_day = 1
#         print("\nGraph and Time Parameters: \n")
#         print("Number of Nodes: .......... " + str(n))
#         print("n days: ................... " + str(n_days))
#
#         print("\nEpidemic Parameters: \n")
#         print("Beta: ..................... " + str(beta))
#         print("Sigma: .................... " + str(sigma))
#         print("Gamma: .................... " + str(gamma))
#         print("Fract Infected: ........... " + str(fr_inf))
#         print()
#
#     # gamma = 1 - numpy.nextafter(1., 0.)
#     # sigma = 50  # sys.maxsize #genero numero molto grande
#     n_inf = int(fr_inf * n)
#
#     flush_structures()
#     initialize()
#     if initial_seed != 0:
#         graph = gm.nx.erdos_renyi_graph(n, 0.05, seed=rg1)
#     else:
#         graph = gm.nx.erdos_renyi_graph(n, 0.05)
#     initialize_infected()
#     interval_tot = n_days * step_p_day
#     sim_SIR(graph, 0, interval_tot)
#     print_SIR_count()
#     plot_SIR_result(validation_type)
#     flush_structures()
#     initialize()
#     initialize_infected()
#     sim_SEIR(graph, 0, interval_tot)
#     print_SEIR_count()
#     plot_SEIR_result(validation_type)


def first_validation(validation_type):
    global n_days
    global n
    global beta
    global gamma
    global sigma
    global n_inf
    global step_p_day
    global people_tot
    global initial_seed

    with open("config_files/" + str(validation_type) + "_input.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        n = data["n_nodes"]  # number of total node
        n_days = data["n_days"]  # number of days
        n_days = data["n_days"]
        beta = data["beta"]
        sigma = data["sigma"]
        gamma = data["gamma"]
        fr_inf = data["fr_inf"]
        initial_seed = data["seed"]
        step_p_day = 1
        print("\nGraph and Time Parameters: \n")
        print("Number of Nodes: .......... " + str(n))
        print("n days: ................... " + str(n_days))

        print("\nEpidemic Parameters: \n")
        print("Beta: ..................... " + str(beta))
        print("Sigma: .................... " + str(sigma))
        print("Gamma: .................... " + str(gamma))
        print("Fract Infected: ........... " + str(fr_inf))
        print()

    # gamma = 1 - numpy.nextafter(1., 0.)
    # sigma = 50  # sys.maxsize #genero numero molto grande
    n_inf = int(fr_inf * n)
    initialize_sir()
    if initial_seed != 0:
        graph = gm.nx.erdos_renyi_graph(n, 0.05, seed=rg1)
    else:
        graph = gm.nx.erdos_renyi_graph(n, 0.05)

    interval_tot = n_days * step_p_day
    sim_sir(graph, 0, interval_tot)
    print_SIR_count()
    plot_SIR_result(validation_type)
    flush_structures()
    initialize_seir()
    sim_seir(graph, 0, interval_tot)
    print_SEIR_count()
    plot_SEIR_result(validation_type)


def second_validation(graphic_name):
    global n_days
    global n
    global beta
    global gamma
    global sigma
    global n_inf
    global s_list
    global i_list
    global r_list
    global step_p_day
    global people_tot
    global initial_seed

    with open("config_files/" + str(graphic_name) + "_input.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        n = data["n_nodes"]  # number of total node
        n_days = data["n_days"]  # number of days
        n_days = data["n_days"]
        beta = data["beta"]
        sigma = data["sigma"]
        gamma = data["gamma"]
        fr_inf = data["fr_inf"]
        step_p_day = data["step_p_day"]
        initial_seed = data["seed"]
        print("\nGraph and Time Parameters: \n")
        print("Number of Nodes: .......... " + str(n))
        print("n days: ................... " + str(n_days))
        print("step per day............... " + str(step_p_day))

        print("\nEpidemic Parameters: \n")
        print("Beta: ..................... " + str(beta))
        print("Sigma: .................... " + str(sigma))
        print("Gamma: .................... " + str(gamma))
        print("Fract Infected: ........... " + str(fr_inf))
        print()

    n_inf = int(fr_inf * n)
    if initial_seed != 0:
        graph = gm.nx.erdos_renyi_graph(n, 0.05, seed=rg1)
    else:
        graph = gm.nx.erdos_renyi_graph(n, 0.05)
    initialize_sir()
    interval_tot = n_days * step_p_day
    sim_sir(graph, 0, interval_tot)
    print_SIR_count()
    plot_SIR_result(graphic_name + "_" + str(step_p_day) + "_steps")
    flush_structures()
    with open("config_files/" + str(graphic_name) + "_input.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        n = data["n_nodes"]  # number of total node
        n_days = data["n_days"]  # number of days
        n_days = data["n_days"]
        beta = data["beta"]
        sigma = data["sigma"]
        gamma = data["gamma"]
        fr_inf = data["fr_inf"]
        step_p_day = data["step_p_day"]
        initial_seed = data["seed"]
        print("\nGraph and Time Parameters: \n")
        print("Number of Nodes: .......... " + str(n))
        print("n days: ................... " + str(n_days))

        print("\nEpidemic Parameters: \n")
        print("Beta: ..................... " + str(beta))
        print("Sigma: .................... " + str(sigma))
        print("Gamma: .................... " + str(gamma))
        print("Fract Infected: ........... " + str(fr_inf))
        print()
    step_p_day = 1
    initialize_sir()
    interval_tot = n_days
    sim_sir(graph, 0, interval_tot)
    print_SIR_count()
    plot_SIR_result(graphic_name + "_1_step")


def flush_structures():
    global sir_list
    global res_time_list

    global s_t
    global e_t
    global i_t
    global r_t

    sir_list = []
    res_time_list = []

    s_t = []
    e_t = []
    i_t = []
    r_t = []


# def flush_structures():
#     global s_list
#     global e_list
#     global i_list
#     global r_list
#
#     global s_t
#     global e_t
#     global i_t
#     global r_t
#
#     s_list = []
#     e_list = []
#     i_list = []
#     r_list = []
#
#     s_t = []
#     e_t = []
#     i_t = []
#     r_t = []
#
#     gc.get_stats()


# def initialize_old():
#     global fr_inf
#     global step_p_day
#     global people_tot
#     global n_inf
#
#     global s_list
#     global e_list
#     global i_list
#     global r_list
#     global seed
#     global rg1
#
#     rg1 = set_random_stream()
#     people_tot = [elm for elm in range(0, n)]
#     rg1.shuffle(people_tot)
#     initial_i = [el for el in people_tot[:n_inf]]
#     i_list = [[el, 0] for el in initial_i]
#     for elm in people_tot:
#         if is_infected(elm) == -1:
#             s_list.append(elm)
#     e_list = []
#     r_list = []
#     # print(n_app_users)
#     # input()
#     rg1.shuffle(people_tot)
#     traced_people = [False for elem in range(0, n)]
#     people_status = [0 for elem in range(0, n)]
#     for pr in people_tot[:n_app_users]:
#         traced_people[pr] = True
#     print(fr_symptomatic)


def initialize_tracing():
    global people_tot
    global seir_list
    global res_time_list
    global app_people
    global contact_list
    global gamma
    global sigma
    global beta

    global rg1

    gamma = gamma * (1 / step_p_day)
    sigma = sigma * (1 / step_p_day)
    beta = beta * (1 / step_p_day)

    rg1 = set_random_stream()
    people_tot = [elm for elm in range(0, n)]
    seir_list = [0 for elm in range(0, n)]
    res_time_list = [0 for elm in range(0, n)]
    app_people = [False for elem in range(0, n)]
    contact_list = [[] for elem in range(0, n)]

    rg1.shuffle(people_tot)
    initial_i = [el for el in people_tot[:n_inf]]
    rg1.shuffle(people_tot)
    for pr in people_tot[:n_app_users]:
        app_people[pr] = True
    for inf in initial_i:
        set_contagion(inf)


def initialize_sir():
    global fr_inf
    global step_p_day
    global people_tot
    global n_inf
    global sir_list
    global res_time_list
    global gamma
    global sigma
    global beta

    global seed
    global rg1

    gamma = gamma * (1 / step_p_day)
    beta = beta * (1 / step_p_day)

    rg1 = set_random_stream()
    people_tot = [elm for elm in range(0, n)]
    sir_list = [0 for elm in range(0, n)]
    res_time_list = [0 for elm in range(0, n)]

    rg1.shuffle(people_tot)
    initial_i = [el for el in people_tot[:n_inf]]
    for inf in initial_i:
        sir_list[inf] = 1
        res_time_list[inf] = rg1.expovariate(gamma)


def initialize_seir():
    global fr_inf
    global step_p_day
    global people_tot
    global n_inf
    global seir_list
    global res_time_list
    global gamma
    global sigma
    global beta

    global seed
    global rg1

    gamma = gamma * (1 / step_p_day)
    sigma = sigma * (1 / step_p_day)
    beta = beta * (1 / step_p_day)

    rg1 = set_random_stream()
    people_tot = [elm for elm in range(0, n)]
    seir_list = [0 for elm in range(0, n)]
    res_time_list = [0 for elm in range(0, n)]

    rg1.shuffle(people_tot)
    initial_i = [el for el in people_tot[:n_inf]]
    for inf in initial_i:
        seir_list[inf] = 2
        res_time_list[inf] = rg1.expovariate(gamma)


def simulate_tracing():
    global fam_graph
    global office_school_graph
    global transp_graph

    initialize_tracing()
    start_time = time.time()
    fam_graph = create_families_network()
    update_contact_list(fam_graph, 1)
    office_school_graph = create_school_work_network()
    # loc_contact(office_school_graph, 2)
    update_contact_list(office_school_graph, 1)
    # update_contact_list(office_school_graph, 1)
    transp_graph = create_transport_network()
    # gm.print_graph_with_labels_and_neighb(transp_graph)
    update_contact_list(transp_graph, 1)
    # print_contact_list()
    # input()
    end_time = time.time()
    duration = round((end_time - start_time), 3)
    gc.collect()
    print("Duration Graph Creation: " + str(duration) + " Seconds")
    start_time = time.time()
    for day in range(0, n_days):
        print("day " + str(day))
        # HOME
        end_1 = day + n_step_home
        sim_seir_tracing(fam_graph, day, end_1)
        # TRANSP
        end_2 = int(end_1 + (n_step_transp / 2))
        transp_graph = create_transport_network()
        # gm.print_graph_with_labels_and_neighb(transp_graph)
        update_contact_list(transp_graph, 1)
        sim_seir_tracing(transp_graph, end_1, end_2)
        # WORK
        end_3 = end_2 + n_step_work
        sim_seir_tracing(office_school_graph, end_2, end_3)
        # TRANSP
        transp_graph = create_transport_network()
        # gm.print_graph_with_labels_and_neighb(transp_graph)
        update_contact_list(transp_graph, 1)
        sim_seir_tracing(transp_graph, end_3, int(end_3 + (n_step_transp / 2)))

    end_time = time.time()
    duration = round((end_time - start_time), 3)
    print("Duration SEIR Simulation: " + str(duration) + " Seconds")

    plot_SEIR_result("pt1")
    print_SEIR_count()


def simulate():
    initialize_seir()
    print_seir_list()
    print(pr_notification)
    print(pr_diagnosis)
    print(app_people)
    input()
    start_time = time.time()
    fam_graph = create_families_network()
    office_school_graph = create_school_work_network()
    transp_graph = create_transport_network()
    end_time = time.time()
    duration = round((end_time - start_time), 3)
    print("Duration Graph Creation: " + str(duration) + " Seconds")
    start_time = time.time()
    for day in range(0, n_days):
        print("day " + str(day))
        # HOME
        end_1 = day + n_step_home
        sim_seir(fam_graph, day, end_1)
        # TRANSP
        end_2 = int(end_1 + (n_step_transp / 2))
        sim_seir(transp_graph, end_1, end_2)
        # WORK
        end_3 = end_2 + n_step_work
        sim_seir(office_school_graph, end_2, end_3)
        # WORK
        sim_seir(transp_graph, end_3, int(end_3 + (n_step_transp / 2)))

    end_time = time.time()
    duration = round((end_time - start_time), 3)
    print("Duration SEIR Simulation: " + str(duration) + " Seconds")

    plot_SEIR_result("prova1")
    print_SEIR_count()
    # flush_structures()
    # initialize()
    # initialize_infected()
    # sim_SEIR(gm.nx.erdos_renyi_graph(500, 0.13), 0, 50)
    # print_SEIR_count()
    # input()
    # fam_gr.clear()
    # for elem in graphs:
    #     gm.print_graph_with_labels_and_neighb(elem)


def print_contact_list():
    index = 0
    for elem in contact_list:
        print("Nodo " + str(index) + ": " + str(elem))
        index += 1


def get_statistic_seir_tracing():
    count = [0, 0, 0, 0, 0, 0]  # n_s, n_e, n_i, n_r, n_isol, n_q
    # print(seir_list)
    for elem in seir_list:
        count[elem] += 1
    return count


def get_statistic_seir():
    count = [0, 0, 0, 0]  # n_s, n_e, n_i, n_r
    # print(seir_list)
    for elem in seir_list:
        count[elem] += 1
    return count


def get_statistic_sir():
    count = [0, 0, 0]  # n_s, n_i, n_r
    # print(sir_list)
    for elem in sir_list:
        count[elem] += 1
    return count


# def loc_contact(inf):
#     transp_graph
#     node_list = dict(office_school_graph.nodes(data="school"))
#     if node_list[inf][0] == "public_transport":
#         r = rg1.uniform(0.0,1.0)
#         if r<pr_notification:
#             seir_list[inf] = 5
#             res_time_list[inf] = n_days_quar * step_p_day

def set_contagion(inf):
    global seir_list
    global res_time_list
    r1 = rg1.uniform(0.0, 1.0)
    if r1 < pr_diagnosis:
        seir_list[inf] = 4
        res_time_list[inf] = n_days_isol * step_p_day
        for elem in contact_list[inf]:
            if app_people[elem[0]]:
                seir_list[elem[0]] = 5
                res_time_list[elem[0]] = n_days_quar * step_p_day
        if app_people[inf]:
            stop = int(pr_notification * len(contact_list[inf]))
            for index in range(0, stop):
                seir_list[contact_list[inf][index][0]] = 5
                res_time_list[contact_list[inf][index][0]] = n_days_quar * step_p_day
    else:
        seir_list[inf] = 2
        res_time_list[inf] = rg1.expovariate(gamma)


def update_node_contacts(node, list_2, timestamp):
    global contact_list
    i = 0
    j = 0
    res = []
    while i < len(contact_list[node]) and j < len(list_2):
        if contact_list[node][i][0] < list_2[j]:
            res.append([contact_list[node][i][0], contact_list[node][i][1]])
            i += 1
        elif contact_list[node][i][0] == list_2[j]:  # elimino doppioni
            res.append([contact_list[node][i][0], timestamp])
            j += 1
            i += 1
        else:
            res.append([list_2[j], timestamp])
            j += 1
    # Elementi rimasti
    while i < len(contact_list[node]):
        res.append([contact_list[node][i][0], contact_list[node][i][1]])
        i += 1
    while j < len(list_2):
        res.append([list_2[j], timestamp])
        j += 1
    contact_list[node].clear()
    contact_list[node] = res


def update_contact_list(graph, timestamp, g_is_sorted=False):
    for elem in graph.nodes(data="graph_name"):
        el = list(elem)
        node_id = el[0]
        g_name = el[1][0]
        # print(node_id)
        # print(g_name)
        # input()
        if g_is_sorted:
            nbs = list(gm.nx.neighbors(graph, node_id))
        else:
            nbs = sorted(list(gm.nx.neighbors(graph, node_id)))
        update_node_contacts(node_id, nbs, timestamp)
        # print(list(gm.nx.neighbors(graph, elem)))
    # gc.collect()


def sim_sir(graph, start_t, end_t):
    global s_t
    global i_t
    global r_t

    for step in range(start_t, end_t):
        [n_s, n_i, n_r] = get_statistic_sir()
        s_t.append(n_s)
        i_t.append(n_i)
        r_t.append(n_r)

        for index in range(0, len(res_time_list)):
            if res_time_list[index] > 0.5:
                res_time_list[index] -= 1
                if sir_list[index] == 1:
                    ngbs = graph.neighbors(index)
                    for ngb in ngbs:
                        if sir_list[ngb] == 0:
                            r = rg1.uniform(0.0, 1.0)
                            if r < beta:
                                # S --> I
                                res_time_list[ngb] = rg1.expovariate(gamma)
                                sir_list[ngb] = 1
            elif sir_list[index] == 1:
                # I --> R
                res_time_list[index] = 0
                sir_list[index] = 2


def sim_seir(graph, start_t, end_t):
    global s_t
    global e_t
    global i_t
    global r_t

    global seir_list
    global res_time_list

    for step in range(start_t, end_t):
        [n_s, n_e, n_i, n_r] = get_statistic_seir()
        s_t.append(n_s)
        e_t.append(n_e)
        i_t.append(n_i)
        r_t.append(n_r)

        for index in range(0, len(res_time_list)):
            if res_time_list[index] > 0.5:
                res_time_list[index] -= 1
                if seir_list[index] == 2:  # index is I
                    ngbs = graph.neighbors(index)
                    for ngb in ngbs:
                        if seir_list[ngb] == 0:
                            r = rg1.uniform(0.0, 1.0)
                            # S --> E
                            if r < beta:
                                res_time_list[ngb] = rg1.expovariate(sigma)
                                seir_list[ngb] = 1
            elif seir_list[index] == 1:
                # E --> I
                res_time_list[index] = rg1.expovariate(gamma)
                seir_list[index] = 2
            elif seir_list[index] == 2:
                res_time_list[index] = 0
                seir_list[index] = 3


def sim_seir_tracing(graph, start_t, end_t):
    global s_t
    global e_t
    global i_t
    global r_t

    global seir_list
    global res_time_list

    for step in range(start_t, end_t):
        [n_s, n_e, n_i, n_r, n_is, n_q] = get_statistic_seir_tracing()
        s_t.append(n_s)
        e_t.append(n_e)
        i_t.append(n_i)
        r_t.append(n_r)

        for index in range(0, len(res_time_list)):
            if res_time_list[index] > 0.5:
                res_time_list[index] -= 1
                if seir_list[index] == 2:  # index is I
                    ngbs = graph.neighbors(index)
                    for ngb in ngbs:
                        if seir_list[ngb] == 0:
                            r = rg1.uniform(0.0, 1.0)
                            # S --> E
                            if r < beta:
                                res_time_list[ngb] = rg1.expovariate(sigma)
                                seir_list[ngb] = 1
            elif seir_list[index] == 1:
                # E --> I
                set_contagion(index)
            elif seir_list[index] == 2:
                # I --> R
                res_time_list[index] = 0
                seir_list[index] = 3
            elif seir_list[index] == 4 or seir_list[index] == 5:
                res_time_list[index] = 0
                seir_list[index] = 3


def sim_SEIR_old(graph, start_t, end_t):
    global s_list
    global e_list
    global i_list
    global r_list
    global s_t
    global e_t
    global i_t
    global r_t

    gamma = gamma * (1 / step_p_day)
    sigma = sigma * (1 / step_p_day)
    #beta = beta * (1 / step_p_day)

    for step in range(start_t, end_t):

        s_t.append(len(s_list))
        e_t.append(len(e_list))
        i_t.append(len(i_list))
        r_t.append(len(r_list))

        for index in range(len(i_list) - 1, -1, -1):
            # I --> R
            if i_list[index][1] <= 0.5:
                r_list.append(i_list[index][0])
                i_list.remove(i_list[index])
            else:
                i_list[index][1] -= 1
        # print("Prima: " + str(e_list))
        for index in range(len(e_list) - 1, -1, -1):
            # E --> I
            if e_list[index][1] <= 0.5:
                duration_gamma = rg1.expovariate(gamma)
                i_list.append([e_list[index][0], duration_gamma])
                e_list.remove(e_list[index])
            else:
                e_list[index][1] -= 1
        for index in range(0, len(i_list)):
            ngbs = graph.neighbors(i_list[index][0])
            for ngb in ngbs:
                if ngb in s_list:
                    r = rg1.uniform(0.0, 1.0)
                    # S --> E
                    if r < beta:
                        duration_sigma = rg1.expovariate(sigma)
                        e_list.append([ngb, duration_sigma])
                        s_list.remove(ngb)


def sim_SIR_old(graph, start_t, end_t):
    global s_list
    global i_list
    global r_list
    global s_t
    global i_t
    global r_t
    global gamma
    global beta
    global step_p_day

    gamma = gamma * (1 / step_p_day)
    beta = beta * (1 / step_p_day)

    for step in range(start_t, end_t):
        # if (step % step_p_day) == 0:
        s_t.append(len(s_list))
        e_t.append(len(e_list))
        i_t.append(len(i_list))
        r_t.append(len(r_list))
        # I --> R
        for index in range(len(i_list) - 1, -1, -1):
            # if infect[0] in part:
            if i_list[index][1] <= 0.5:  # abbiamo superato la durata dell'infezione generata
                r_list.append(i_list[index][0])
                i_list.remove(i_list[index])
            else:
                i_list[index][1] -= 1

        for index in range(0, len(i_list)):
            ngbs = graph.neighbors(i_list[index][0])
            for ngb in ngbs:
                if ngb in s_list:
                    r = rg1.uniform(0.0, 1.0)
                    if r < beta:  # CONTAGIO
                        duration_gamma = rg1.expovariate(gamma)
                        i_list.append([ngb, duration_gamma])
                        s_list.remove(ngb)


def print_SEIR_count():
    print("\nCount SEIR: ")
    print("s_t: " + str(s_t))
    print("e_t: " + str(e_t))
    print("i_t: " + str(i_t))
    print("r_t: " + str(r_t))


def print_SIR_count():
    print("\nCount SIR: ")
    print("s_t: " + str(s_t))
    print("i_t: " + str(i_t))
    print("r_t: " + str(r_t))


def print_sir_list():
    print("\nLists: ")
    print("sir_list: " + str(sir_list))
    print("res_time_list: " + str(res_time_list))


def print_seir_list():
    print("\nLists: ")
    print("seir_list: " + str(seir_list))
    print("res_time_list: " + str(res_time_list))


def plot_SIR_result(filename):
    time = []
    for i in range(0, len(s_t)):
        time.append(i / step_p_day)
    # list_time_new = np.linspace(min(time), max(time), 1000)
    plt.plot(time, s_t, color='blue')
    plt.plot(time, i_t, color='red')
    plt.plot(time, r_t, color='yellow')
    # x_int = range(min(time), math.ceil(max(time)) + 1)
    # plt.xticks(x_int) # per avere interi nelle ascisse
    plt.title('Simulation Result', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('', fontsize=14)
    plt.grid(True)

    # legend
    blue_patch = mpatches.Patch(color='blue', label='Susceptible')
    red_patch = mpatches.Patch(color='red', label='Infected')
    yellow_patch = mpatches.Patch(color='yellow', label='Reduced')
    plt.legend(handles=[blue_patch, red_patch, yellow_patch])

    plt.savefig("img/" + str(filename) + "_SIR.png")
    plt.close()


def plot_SEIR_result(filename):
    time = []
    for i in range(0, len(s_t)):
        time.append(i / step_p_day)
    # list_time_new = np.linspace(min(time), max(time), 1000)
    plt.plot(time, s_t, color='blue')
    plt.plot(time, e_t, color='orange')
    plt.plot(time, i_t, color='red')
    plt.plot(time, r_t, color='yellow')
    # x_int = range(min(time), math.ceil(max(time)) + 1)
    # plt.xticks(x_int) # per avere interi nelle ascisse
    plt.title('Simulation Result - SEIR', fontsize=14)
    plt.xlabel('Time (gg)', fontsize=14, )
    plt.ylabel('', fontsize=14)
    plt.grid(True)

    # legend
    blue_patch = mpatches.Patch(color='blue', label='Susceptible')
    orange_patch = mpatches.Patch(color='orange', label='Exposed')
    red_patch = mpatches.Patch(color='red', label='Infected')
    yellow_patch = mpatches.Patch(color='yellow', label='Reduced')
    plt.legend(handles=[blue_patch, orange_patch, red_patch, yellow_patch])

    plt.savefig("img/" + str(filename) + "_SEIR.png")
    plt.close()


# def count_SEIR():
#     global s_t
#     global e_t
#     global i_t
#     global r_t
#     s_t.append(len(s_list))
#     e_t.append(len(e_list))
#     i_t.append(len(i_list))
#     r_t.append(len(r_list))


def epidemic(graph, n_days):
    global beta
    mu = gamma

    # scelgo random gli infetti iniziali
    list_nodes = list(graph.nodes())
    rg1.shuffle(list_nodes)
    initial_infections = list_nodes[0:n_inf]
    sim = eon.fast_SIR(graph, beta, mu, initial_infecteds=initial_infections, tmax=n_days, return_full_data=True)
    t = sim.t()
    S = sim.S()  # numero suscettibili ad ogni istante
    I = sim.I()  # numero infetti ad ogni istante
    R = sim.R()  # numero rimossi ad ogni istante

    r_per = R[-1] / len(graph.nodes()) * 100
    s_per = S[-1] / len(graph.nodes()) * 100
    i_per = I[-1] / len(graph.nodes()) * 100

    # Print Result
    plt.plot(t, S, label='S')
    plt.plot(t, I, label='I')
    plt.plot(t, R, label='R')
    plt.legend()
    plt.savefig('img/comparison_EON_beta_' + str(beta) + ' ' + 'mu_' + str(mu) + '_SIR.png')
    plt.close()

    # print('animation...')
    # ani = sim.animate(ts_plots=['I', 'SIR'], node_size=4)
    # writer = animation.PillowWriter('fps=2')
    # # ani.save('SIR.mp4',writer=Writer,fps=5, extra_args=['-vcodec', 'libx264'])
    # ani.save('compare_EON_beta_' + str(beta) + ' ' + 'mu_' + str(mu) + ' SIR.gif')
    plt.close()


# def initialize_infected():
#     for infect in i_list:
#         infect[1] = rg1.expovariate(gamma)
#
#
# def is_infected(elem):
#     ctrl = True
#     i = 0
#     while ctrl and i < len(i_list):
#         ctrl = not (elem == i_list[i][0])
#         i += 1
#     if ctrl:
#         return -1
#     return i - 1
#
#
# def is_exposed(elem):
#     ctrl = True
#     i = 0
#     while ctrl and i < len(e_list):
#         ctrl = not (elem == e_list[i][0])
#         i += 1
#     if ctrl:
#         return -1
#     return i - 1


def parse_input_file():
    global beta
    global sigma
    global gamma

    global n
    global n_inf
    global n_stud
    global n_employs
    global n_app_users
    global fr_symptomatic

    global min_family_size
    global max_familiy_size
    global min_school_size
    global max_school_size
    global min_office_size
    global max_office_size
    global min_transp_size
    global max_transp_size

    global step_p_day
    global n_step_home
    global n_step_work
    global n_step_transp
    global n_days
    global n_days_isol
    global n_days_quar
    global initial_seed

    global pr_diagnosis
    global pr_notification

    with open("config_files/graph_and_time_parameters.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        n = data["n_nodes"]  # number of total node
        n_days = data["n_days"]  # number of days
        # step_p_day = data["step_p_day"]
        n_step_home = data["n_step_home"]
        n_step_work = data["n_step_work"]
        n_step_transp = data["n_step_transp"]
        n_stud = int(data["fr_students"] * n)
        n_employs = int(data["fr_employees"] * n)
        min_family_size = data["min_family_size"]
        max_familiy_size = data["max_familiy_size"]
        min_school_size = data["min_school_size"]
        max_school_size = data["max_school_size"]
        min_office_size = data["min_office_size"]
        max_office_size = data["max_office_size"]
        min_transp_size = data["min_transp_size"]
        max_transp_size = data["max_transp_size"]
        n_app_users = int(data["fr_app_users"] * n)
        initial_seed = data["seed"]
        pr_diagnosis = data["pr_diagnosis"]
        pr_notification = data["pr_notification"]
        step_p_day = n_step_home + n_step_work + n_step_transp

        print("\nGraph and Time Parameters: \n")
        print("Number of Nodes: .......... " + str(n))
        print("n days: ................... " + str(n_days))
        print("Step per day............... " + str(step_p_day))
        print("Step spent at home......... " + str(n_step_home))
        print("Step spent at work......... " + str(n_step_work))
        print("Number of Students: ....... " + str(n_stud))
        print("Number of Employee: ....... " + str(n_employs))
        print()
        print("Family size: .............. " + str(min_family_size) + " - " + str(max_familiy_size))
        print("School size: .............. " + str(min_school_size) + " - " + str(max_school_size))
        print("Prob Diagnosi.............. " + str(pr_diagnosis))
        print("Prob Ricezione Notifica.... " + str(pr_notification))

    with open("config_files/epidemic_parameters.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        beta = data["beta"]  # probability of contagion
        sigma = data["sigma"]  # transition rate from E to I
        gamma = data["gamma"]  # transition rate from I to R
        n_inf = int(data["fr_inf"] * n)  # number of initial infected
        fr_symptomatic = data["fr_symptomatic"]
        n_days_isol = data["n_days_isol"]
        n_days_quar = data["n_days_quar"]

        print("\nEpidemic Parameters: \n")
        print("Beta: ..................... " + str(beta))
        print("Sigma: .................... " + str(sigma))
        print("Gamma: .................... " + str(gamma))
        print("Number Initial Infected: .. " + str(n_inf))
        print()


if __name__ == '__main__':

    if sys.argv[1] == "validation_1":
        print(sys.argv[1])
        first_validation(sys.argv[1])
    elif sys.argv[1] == "validation_2":
        print(sys.argv[1])
        first_validation(sys.argv[1])
    elif sys.argv[1] == "validation_3":
        print(sys.argv[1])
        second_validation(sys.argv[1])
    elif sys.argv[1] == "tracing":
        parse_input_file()
        simulate_tracing()
    elif sys.argv[1] == "write_res":
        fm.clear_csv()
        s_t = [3, 5, 8]
        e_t = [1, 1, 1]
        i_t = [9, 9, 9]
        r_t = [10, 1, 1]
        fm.write_csv(s_t, e_t, i_t, r_t, 1)
        s_t = [4, 5, 9]
        e_t = [1, 1, 1]
        i_t = [3, 5, 8]
        r_t = [10, 1, 1]
        fm.write_csv(s_t, e_t, i_t, r_t, 2)
        s_t = [5, 5, 8]
        e_t = [1, 5, 1]
        i_t = [9, 9, 9]
        r_t = [10, 1, 1]
        fm.write_csv(s_t, e_t, i_t, r_t, 3)
        [a, b, c, d] = fm.calculate_average_from_csv()
        print(a)
        print(b)
        print(c)
        print(d)
    elif sys.argv[1] == "simulate":
        parse_input_file()
        simulate()
    elif sys.argv[1] == "compare":
        compare_with_EON("comparison")
    elif sys.argv[1] == "random":
        rg1 = random.Random()
        rg1.seed(2)
        print(rg1.random())
        rg1.seed(1)
        print(rg1.random())
        rg1 = random.Random(1)
        print(rg1.random())
    elif sys.argv[1] == "test":

        graph = gm.nx.erdos_renyi_graph(15, 0.1)
        contact_list = [[] for elem in range(0, 15)]
        update_contact_list(graph, 1)
        graph = gm.nx.erdos_renyi_graph(15, 0.1)
        update_contact_list(graph, 2)

        print_contact_list()


    else:
        parse_input_file()
