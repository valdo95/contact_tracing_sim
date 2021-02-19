from multiprocessing import Process, Lock
import graph_manager as gm
import file_manager as fm
import sys
import gc
import random
# import EoN as eon  # not used
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from itertools import islice
import time
import yaml

home_step = 0  # step for days
work_step = 0  # step for days
n_days = 0  # number of days
n_days_quar = 0  # quarantine length
n_days_isol = 0  # isolation length
beta = 0  # probability of contagion
sigma = 0  # transition rate from E to I
gamma = 0  # transition rate from I to R
eta = 0  # transition rate from Q-EI to Is
fr_inf = 0  # number of initial infected
step_p_day = 0  # number of step per day
n_step_home = 0  # number of step at home
n_step_work = 0  # number of step at work/school
n_step_transp = 0  # number of step on public transport
fr_station_user = 0  # fraction of people that use train station
fr_symptomatic = 0  # fraction of symptomatic
initial_seed = 0
n_proc = 0  # Number of free core that can run tihs code
start_contagion = 0  # time of first contagion
rg1 = None
fam_graph = None
office_school_graph = None
transp_graph = None
office_graph = None
school_graph = None
# COUNTS
n = 0  # number of total node
n_inf = 0  # number of initial infected
n_stud = 0  # number of student
n_employs = 0  # number of employees
n_app_users = 0  # number of people with the app
n_s = 0  # number of simulation
abs = False  # if user want abs graph result

pr_diagnosis = 0  # Prob of been diagnosed
pr_notification = 0  # Prob of receive a Notification

window_size = 0  # Size of the contact window

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

seir_list = []  # 0-->S, 1-->E, 2-->I, 3-->R, 4-->Isol., 5-->Quarantine S, 6-->Quarantine EI
sir_list = []  # 0-->S, 1-->I, 2-->R
res_time_list = []  # res. times (if the node is I or E)
s_t = []  # number of susceptible for each step (e. g. step_p_day = 10 -> step = 1 / 10)
e_t = []  # number of exposed for each step
i_t = []  # number of infected for each step
r_t = []  # number of recovered/isolated/dead for each step
is_t = []  # number of isolated nodes
qs_t = []  # number of quarantined s for each step
qei_t = []  # number of quarantined ei for each step

people_tot = []  # array of nodes
app_people = []  # One entry for each person: The values are True if the person use the app
contact_list = []  # [ngb, timestamp1, timestamp2] contact list
contact_matrix = []  # [ngb, timestamp1, timestamp2] contact list
# people_status = []  # One entry for each person: 0 -> S, value > 0 --> value represent the residue time in quarantine o isolation

# commuter_partitions = []  # list of list of people that use one specific station
public_transport_users = []  # list of list of people that use one specific public_transport/bus

graph_ext = []  # "Transport", "School","Office","Families"
# GRAPH DENSITY
school_density = 0
office_density = 0
transport_density = 0

office_partion = []
school_partition = []
transp_partition = []

p_name = 0  # process name
tracing = False  # False --> SEIR, True --> SEIR +tracing
is_sparse = False
fr_far_contacts = False
# Initial Seed for MultiProc Sim: they have been generated from random.org (I didn't use python rng to avoid overlapping)
initial_seeds = [5628732, 6653, 369944, 980321, 930450, 6879238, 1260548, 4566454, 8015103, 2418865, 7687303, 2803321,
                 278620, 3564, 575677, 706267, 49141, 935732, 277247, 694306]


def set_random_stream(proc_seed=0):
    if initial_seed == 0:
        return random.Random()
    elif proc_seed == 0:
        return random.Random(initial_seed)
    else:
        return random.Random(proc_seed)


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
    res = gm.nx.union_all(temp_graphs)
    res.name = "Families"
    return res


def create_transport_network():
    rg1.shuffle(people_tot)
    transp_part = list(generate_partitions(people_tot, min_transp_size, max_transp_size))
    # print("families: " + str(families))
    temp_graphs = []
    for bus in transp_part:
        temp_graphs.append(gm.create_public_transport_graph(bus, rg=rg1, density=transport_density))
    res = gm.nx.union_all(temp_graphs)
    res.name = "Transport"
    return res


def create_school_work_network():
    rg1.shuffle(people_tot)
    office_partitions = list(generate_partitions(people_tot[:n_stud], min_office_size, max_office_size))
    school_partitions = list(
        generate_partitions(people_tot[n_stud:(n_stud + n_employs)], min_school_size, max_school_size))
    # print("office partitions: " + str(office_partitions))
    # print("school partitions: " + str(school_partitions))
    temp_graphs = []
    for office in office_partitions:
        temp_graphs.append(gm.create_office_graph(office, density=office_density, rg=rg1))
    office_graph = gm.nx.union_all(temp_graphs)
    office_graph.name = "Office"
    temp_graphs = []
    for school in school_partitions:
        temp_graphs.append(gm.create_school_graph(school, density=school_density, rg=rg1))
    school_graph = gm.nx.union_all(temp_graphs)
    school_graph.name = "School"
    return office_graph, school_graph


# def compare_with_EON(comparison_type):
#     global n
#     global n_days
#     global beta
#     global gamma
#     global n_inf
#     global step_p_day
#     with open("config_files/" + str(comparison_type) + "_input.yaml", 'r') as stream:
#         data = yaml.safe_load(stream)
#         n = data["n_nodes"]  # number of total node
#         n_days = data["n_days"]  # number of days
#         beta = data["beta"]
#         gamma = data["gamma"]
#         fr_inf = data["fr_inf"]
#         step_p_day = data["step_p_day"]
#         print("\nGraph and Time Parameters: \n")
#         print("Number of Nodes: .......... " + str(n))
#         print("n days: ................... " + str(n_days))
#         print("Step per Day............... " + str(step_p_day))
#
#         print("\nEpidemic Parameters: \n")
#         print("Beta: ..................... " + str(beta))
#         print("Gamma: .................... " + str(gamma))
#         print("Fract Infected: ........... " + str(fr_inf))
#         print()
#
#     # gamma = 1 - numpy.nextafter(1., 0.)
#     # sigma = 50  # sys.maxsize #genero numero molto grande
#     n_inf = int(fr_inf * n)
#     initialize_sir()
#     graph = gm.nx.erdos_renyi_graph(n, 0.2)
#     input()
#     print("Start EON Simulation")
#     epidemic(graph, n_days)
#
#     print("End EON Simulation")
#     input()
#     flush_structures()
#     print_SIR_count()
#     initialize_sir()
#     # print(sir_list)
#     # print(res_time_list)
#     # input()
#     print("Start sim_SIR Simulation")
#     sim_SIR_eff(graph, 0, n_days * step_p_day)
#     print("End sim_SIR Simulation")
#     plot_SIR_result("comparison_sim")


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
    sir(graph, 0, interval_tot)
    print_SIR_count()
    plot_SIR_result(validation_type)
    flush_structures()
    initialize_seir()
    seir(graph, 0, interval_tot)
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
    sir(graph, 0, interval_tot)
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
    sir(graph, 0, interval_tot)
    print_SIR_count()
    plot_SIR_result(graphic_name + "_1_step")


def flush_structures():
    global sir_list
    global res_time_list

    global s_t
    global e_t
    global i_t
    global r_t
    global is_t
    global qs_t
    global qei_t

    sir_list = []
    res_time_list = []

    s_t = []
    e_t = []
    i_t = []
    r_t = []
    is_t = []
    qs_t = []
    qei_t = []
    gc.collect()

    # office_graph = None
    # school_graph = None
    # office_school_graph = None
    # transp_graph = None
    # fam_graph = None


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


def delete_old_contacts(curr_time):
    global contact_list
    global contact_matrix
    print("del")
    old_time = curr_time - window_size
    if is_sparse:
        for node_contacts in contact_list:
            for elem in node_contacts:
                if elem[2] < old_time:
                    elem[2] = 0
                if elem[1] < old_time:
                    node_contacts.remove(elem)
    else:
        for line in contact_matrix:
            for index in range(0, len(line)):
                if line[index][0] < old_time or line[index][1] < old_time:
                    line[index] = [-1, -1]


def update_node_contacts(node, list_2, timestamp, check):
    global contact_list
    # global contact_matrix
    i = 0
    j = 0
    res = []
    while i < len(contact_list[node]) and j < len(list_2):
        if contact_list[node][i][0] < list_2[j]:
            # if check:
            res.append([contact_list[node][i][0], contact_list[node][i][1], contact_list[node][i][2]])
            i += 1
        elif contact_list[node][i][0] == list_2[j]:  # doppioni
            if check:
                res.append([contact_list[node][i][0], timestamp, timestamp])
            elif contact_list[node][i][2] >= 0:
                res.append([contact_list[node][i][0], timestamp, contact_list[node][i][2]])
            else:
                res.append([contact_list[node][i][0], timestamp, -1])
            j += 1
            i += 1
        else:  # aggiungo contatto
            if check:
                res.append([list_2[j], timestamp, timestamp])
            else:
                res.append([list_2[j], timestamp, -1])
            j += 1
    # Elementi rimasti
    while i < len(contact_list[node]):
        # if check:
        res.append([contact_list[node][i][0], contact_list[node][i][1], contact_list[node][i][2]])
        i += 1
    while j < len(list_2):
        if check:
            res.append([list_2[j], timestamp, timestamp])
        else:
            res.append([list_2[j], timestamp, -1])
        j += 1
    contact_list[node].clear()
    contact_list[node] = res


def update_matrix(graph, timestamp, node_id, check, second_lev=False):
    global contact_matrix
    # temp = list(gm.nx.neighbors(graph, node_id))
    # ngbs = [temp[index] for index in range(0, int(len(temp) * pr_notification))]
    # temp.clear()
    ngbs = list(gm.nx.neighbors(graph, node_id))
    for ngb in ngbs:
        # print(check)
        # input()
        if check:
            if ngb < node_id:
                contact_matrix[node_id][ngb] = [timestamp, timestamp]
            else:
                contact_matrix[ngb][node_id] = [timestamp, timestamp]
        else:
            if ngb < node_id:
                contact_matrix[node_id][ngb] = [timestamp, -1]
            else:
                contact_matrix[ngb][node_id] = [timestamp, -1]

    if fr_far_contacts != 0:
        temp = list(graph.nodes)
        # rg1.shuffle(temp)
        # print(temp)
        random_list = [temp[index] for index in range(0, int(len(temp) * fr_far_contacts))]
        # print(random_list)
        # input()
        for elem in random_list:
            if elem < node_id:
                contact_matrix[node_id][elem][1] = timestamp
            elif node_id > elem:
                contact_matrix[elem][node_id][1] = timestamp


def update_contacts(graph, timestamp, g_is_sorted=False):
    check = False
    for elem in graph_ext:
        if elem == graph.name:
            check = True
    for node_id in graph.nodes():
        if is_sparse:
            if g_is_sorted:
                nbs = list(gm.nx.neighbors(graph, node_id))
            else:
                nbs = sorted(list(gm.nx.neighbors(graph, node_id)))
            update_node_contacts(node_id, nbs, timestamp, check)
        else:
            update_matrix(graph, timestamp, node_id, check)

        # print(str(node_id)+str(list(gm.nx.neighbors(graph, node_id))))
        # input()


def initialize_constant(seed):
    global gamma
    global sigma
    global beta
    global eta
    global rg1

    gamma = gamma * (1 / step_p_day)
    sigma = sigma * (1 / step_p_day)
    beta = beta  # * (1 / step_p_day)
    eta = eta * (1 / step_p_day)
    # if (step_p_day>96):
    #     beta = beta * (96 / step_p_day)
    rg1 = set_random_stream(seed)


def initialize_tracing():
    global people_tot
    global seir_list
    global res_time_list
    global app_people
    global contact_list
    global contact_matrix

    people_tot = [elm for elm in range(0, n)]
    seir_list = [0 for elm in range(0, n)]
    res_time_list = [0 for elm in range(0, n)]
    app_people = [False for elem in range(0, n)]
    if is_sparse:
        contact_list = [[] for elem in range(0, n)]
    else:
        contact_matrix = [[[-1, -1] for elem in range(0, index)] for index in range(0, n)]  # Matrice triangolare

    rg1.shuffle(people_tot)
    for pr in people_tot[:n_app_users]:
        app_people[pr] = True


def set_fisrt_contagion():
    global seir_list
    global res_time_list
    rg1.shuffle(people_tot)
    initial_i = [el for el in people_tot[:n_inf]]
    for inf in initial_i:
        seir_list[inf] = 2
        res_time_list[inf] = rg1.expovariate(gamma)


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


def sim_seir():
    global office_school_graph
    global office_graph
    global school_graph
    global transp_graph
    global fam_graph

    for day in range(0, n_days):
        print("day " + str(day))
        # HOME
        end_1 = day + n_step_home
        seir(fam_graph, day, end_1)
        # TRANSP
        end_2 = int(end_1 + (n_step_transp / 2))
        transp_graph.clear()
        transp_graph = create_transport_network()
        gc.collect()
        seir(transp_graph, end_1, end_2)
        # WORK
        end_3 = end_2 + n_step_work
        seir(office_school_graph, end_2, end_3)
        # TRANSP
        transp_graph.clear()
        transp_graph = create_transport_network()
        gc.collect()
        seir(transp_graph, end_3, int(end_3 + (n_step_transp / 2)))
        if day == start_contagion:
            set_fisrt_contagion()


def sim_tracing():
    global office_school_graph
    global office_graph
    global school_graph
    global transp_graph
    global fam_graph

    for day in range(0, n_days):
        print("Day " + str(day) + " Process " + str(p_name))
        # HOME
        end_1 = day + n_step_home
        seir_tracing(fam_graph, day, end_1, day=day)
        # TRANSP
        end_2 = int(end_1 + (n_step_transp / 2))
        transp_graph.clear()
        transp_graph = create_transport_network()
        gc.collect()
        seir_tracing(transp_graph, end_1, end_2, day=day)
        # WORK
        end_3 = end_2 + n_step_work
        seir_tracing(office_school_graph, end_2, end_3, day=day)
        # TRANSP
        transp_graph.clear()
        transp_graph = create_transport_network()
        gc.collect()
        seir_tracing(transp_graph, end_3, int(end_3 + (n_step_transp / 2)), day=day)
        if day == start_contagion:
            set_fisrt_contagion()
        if day > 14:
            delete_old_contacts(day)


def sim_tracing_old():
    global office_school_graph
    global office_graph
    global school_graph
    global transp_graph
    global fam_graph

    for day in range(0, n_days):
        print("day " + str(day))
        # HOME
        end_1 = day + n_step_home
        seir_tracing(fam_graph, day, end_1)
        # TRANSP
        end_2 = int(end_1 + (n_step_transp / 2))
        transp_graph.clear()
        transp_graph = create_transport_network()
        update_contacts(transp_graph, day, True)
        gc.collect()
        seir_tracing(transp_graph, end_1, end_2)
        # WORK
        end_3 = end_2 + n_step_work
        seir_tracing(office_school_graph, end_2, end_3)
        # TRANSP
        transp_graph.clear()

        transp_graph = create_transport_network()
        update_contacts(office_graph, day, True)
        update_contacts(school_graph, day, True)
        update_contacts(fam_graph, day)
        gc.collect()
        seir_tracing(transp_graph, end_3, int(end_3 + (n_step_transp / 2)))
        update_contacts(transp_graph, day, True)
        if day % window_size == 0:
            delete_old_contacts(day)


def proc_run_sim(lock, p_seed):
    global office_school_graph
    global office_graph
    global school_graph
    global transp_graph
    global fam_graph
    global p_name

    # parse_input_file()
    p_name = p_seed
    initialize_constant(p_name)
    print("Start Process " + str(p_name))
    print("Tracing " + str(tracing))
    print("N S " + str(n_s))
    for curr_sim in range(0, n_s):
        flush_structures()
        # CREATE GRAPH
        initialize_tracing()
        # start_time = time.time()
        fam_graph = create_families_network()
        update_contacts(fam_graph, 0)
        [office_graph, school_graph] = create_school_work_network()
        office_school_graph = gm.nx.union(office_graph, school_graph)
        transp_graph = create_transport_network()
        gc.collect()
        print("Process " + str(p_name) + "SIMULATION " + str(curr_sim))
        if tracing:
            sim_tracing()
            lock.acquire()
            fm.write_csv_tracing(s_t, e_t, i_t, r_t, is_t, qs_t, qei_t)
            lock.release()
            # plot_seir_tracing(p_name, offset=start_contagion)
            print_tracing_count()

        else:
            sim_seir()
            lock.acquire()
            fm.write_csv_seir(s_t, e_t, i_t, r_t)
            lock.release()
            print_SEIR_count()


def run_sim(tracing):
    global office_school_graph
    global office_graph
    global school_graph
    global transp_graph
    global fam_graph

    initialize_constant(0)
    for curr_sim in range(0, n_s):
        flush_structures()

        # CREATE GRAPH
        initialize_tracing()
        # start_time = time.time()
        fam_graph = create_families_network()
        update_contacts(fam_graph, 0)
        [office_graph, school_graph] = create_school_work_network()
        office_school_graph = gm.nx.union(office_graph, school_graph)
        transp_graph = create_transport_network()
        # end_time = time.time()
        # duration = round((end_time - start_time), 3)
        gc.collect()
        print("SIMULATION " + str(curr_sim))
        if tracing:
            sim_tracing()
            fm.write_csv_tracing(s_t, e_t, i_t, r_t, is_t, qs_t, qei_t)
            print_tracing_count()

        else:
            sim_seir()
            fm.write_csv_seir(s_t, e_t, i_t, r_t)
            print_SEIR_count()


# def simulate():
#     initialize_seir()
#     # print_seir_list()
#     # print(pr_notification)
#     # print(pr_diagnosis)
#     # print(app_people)
#     start_time = time.time()
#     fam_graph = create_families_network()
#     [office_g, school_g] = create_school_work_network()
#     office_school_graph = gm.nx.union(office_g, school_g)
#     transp_graph = create_transport_network()
#     end_time = time.time()
#     duration = round((end_time - start_time), 3)
#     print("Duration Graph Creation: " + str(duration) + " Seconds")
#     start_time = time.time()
#     for day in range(0, n_days):
#         print("day " + str(day))
#         # HOME
#         end_1 = day + n_step_home
#         seir(fam_graph, day, end_1)
#         # TRANSP
#         end_2 = int(end_1 + (n_step_transp / 2))
#         transp_graph = create_transport_network()
#         seir(transp_graph, end_1, end_2)
#         # WORK
#         end_3 = end_2 + n_step_work
#         seir(office_school_graph, end_2, end_3)
#         # TRANSP
#         transp_graph = create_transport_network()
#         seir(transp_graph, end_3, int(end_3 + (n_step_transp / 2)))
#
#     end_time = time.time()
#     duration = round((end_time - start_time), 3)
#     print("Duration SEIR Simulation: " + str(duration) + " Seconds")
#
#     plot_SEIR_result("prova1")
#     print_SEIR_count()
#     # flush_structures()
#     # initialize()
#     # initialize_infected()
#     # sim_SEIR(gm.nx.erdos_renyi_graph(500, 0.13), 0, 50)
#     # print_SEIR_count()
#     # input()
#     # fam_gr.clear()
#     # for elem in graphs:
#     #     gm.print_graph_with_labels_and_neighb(elem)

def get_statistic_seir_tracing():
    count = [0, 0, 0, 0, 0, 0, 0]  # n_s, n_e, n_i, n_r, n_isol, n_qs,nqei
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


def set_contagion(inf):
    global seir_list
    global res_time_list
    r1 = rg1.uniform(0.0, 1.0)
    if r1 < pr_diagnosis:
        # E --> I
        seir_list[inf] = 4
        res_time_list[inf] = n_days_isol * step_p_day

        # GESTIONE NOTIFICHE TRADIZIONALE
        if app_people[inf]:
            if is_sparse:
                # LISTE CONTATTI
                for elem in contact_list[inf]:
                    if app_people[elem[0]]:

                        if seir_list[elem[0]] == 0:
                            # S --> Q-S
                            seir_list[elem[0]] = 5
                            res_time_list[elem[0]] = rg1.expovariate(eta)
                        if seir_list[elem[0]] == 1 or seir_list[elem[0]] == 2:
                            # E or I --> Q-EI
                            seir_list[elem[0]] = 6
                            res_time_list[elem[0]] = n_days_quar * step_p_day

            else:
                # MATRICE CONTATTI
                i = 0
                while i < inf:
                    if contact_matrix[inf][i][0] >= 0 and app_people[i]:

                        if seir_list[i] == 0:
                            # S --> Q-S
                            seir_list[i] = 5
                            res_time_list[i] = rg1.expovariate(eta)  # n_days_quar * step_p_day
                        elif seir_list[i] == 1 or seir_list[i] == 2:
                            # E or I --> Q-EI
                            seir_list[i] = 6
                            res_time_list[i] = n_days_quar * step_p_day
                    i += 1
                i += 1
                while i < n:
                    if contact_matrix[i][inf][0] >= 0 and app_people[i]:
                        if seir_list[i] == 0:
                            # S --> Q-S
                            seir_list[i] = 5
                            res_time_list[i] = rg1.expovariate(eta)  # n_days_quar * step_p_day
                        elif seir_list[i] == 1 or seir_list[i] == 2:
                            # E or I --> Q-EI
                            seir_list[i] = 6
                            res_time_list[i] = n_days_quar * step_p_day
                    i += 1

            # FINE GESTIONE NOTIFICHE TRADIZIONALE

            # LOCALIZZAZIONE
            if is_sparse:
                for index in range(0, len(contact_list[inf])):
                    # r = rg1.uniform(0.0, 1.0)
                    if contact_list[inf][index][
                        2] >= 0 and pr_notification > 0:  # pr_notification > 0 per motivi di effic.
                        r = rg1.uniform(0.0, 1.0)
                        if r < pr_notification:
                            if seir_list[contact_list[inf][index][0]] == 0:
                                # S --> Q-S
                                seir_list[contact_list[inf][index][0]] = 5
                                res_time_list[contact_list[inf][index][0]] = rg1.expovariate(eta)
                            if seir_list[contact_list[inf][index][0]] == 1 or seir_list[
                                contact_list[inf][index][0]] == 2:
                                # E or I --> Q-EI
                                seir_list[contact_list[inf][index][0]] = 6
                                res_time_list[contact_list[inf][index][0]] = n_days_quar * step_p_day

            else:
                i = 0
                while i < inf:
                    if contact_matrix[inf][i][1] >= 0 and pr_notification > 0:
                        r = rg1.uniform(0.0, 1.0)
                        if r < pr_notification:
                            if seir_list[i] == 0:
                                # S --> Q-S
                                seir_list[i] = 5
                                res_time_list[i] = rg1.expovariate(eta)
                            elif seir_list[i] == 1 or seir_list[i] == 2:
                                # E or I --> Q-EI
                                seir_list[i] = 6
                                res_time_list[i] = n_days_quar * step_p_day
                    i += 1
                i += 1
                while i < n:

                    if contact_matrix[i][inf][
                        1] >= 0 and pr_notification > 0:  # pr_notification > 0 per motivi di effic.
                        r = rg1.uniform(0.0, 1.0)
                        if r < pr_notification:
                            # S --> Q-S
                            if seir_list[i] == 0:
                                seir_list[i] = 5
                                res_time_list[i] = rg1.expovariate(eta)  # n_days_quar * step_p_day
                            elif seir_list[i] == 1 or seir_list[i] == 2:
                                # E or I --> Q-EI
                                seir_list[i] = 6
                                res_time_list[i] = n_days_quar * step_p_day
                    i += 1

    else:
        # E --> I
        seir_list[inf] = 2
        res_time_list[inf] = rg1.expovariate(gamma)


def sir(graph, start_t, end_t):
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


def seir(graph, start_t, end_t):
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


def seir_tracing(graph, start_t, end_t, day):
    global s_t
    global e_t
    global i_t
    global r_t
    global is_t
    global qs_t
    global qei_t

    global seir_list
    global res_time_list

    update = 0
    for step in range(start_t, end_t):
        [n_s, n_e, n_i, n_r, n_is, n_qs, n_qei] = get_statistic_seir_tracing()
        s_t.append(n_s)
        e_t.append(n_e)
        i_t.append(n_i)
        r_t.append(n_r)
        is_t.append(n_is)
        qs_t.append(n_qs)
        qei_t.append(n_qei)

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
            elif seir_list[index] == 1 or seir_list[index] == 6:
                # (E --> I or Is) or Q_EI --> Is (+ gestione notifiche)
                set_contagion(index)
            elif seir_list[index] == 2:
                # I --> R
                res_time_list[index] = 0
                seir_list[index] = 3
            elif seir_list[index] == 4:
                # Is --> R
                res_time_list[index] = 0
                seir_list[index] = 3
            elif seir_list[index] == 5:
                # Q_S --> S
                res_time_list[index] = 0
                seir_list[index] = 0

            if update == 0:
                update_contacts(graph, day)
            update += 1


# def sim_SEIR_old(graph, start_t, end_t):
#     global s_list
#     global e_list
#     global i_list
#     global r_list
#     global s_t
#     global e_t
#     global i_t
#     global r_t
#
#     gamma = gamma * (1 / step_p_day)
#     sigma = sigma * (1 / step_p_day)
#     # beta = beta * (1 / step_p_day)
#
#     for step in range(start_t, end_t):
#
#         s_t.append(len(s_list))
#         e_t.append(len(e_list))
#         i_t.append(len(i_list))
#         r_t.append(len(r_list))
#
#         for index in range(len(i_list) - 1, -1, -1):
#             # I --> R
#             if i_list[index][1] <= 0.5:
#                 r_list.append(i_list[index][0])
#                 i_list.remove(i_list[index])
#             else:
#                 i_list[index][1] -= 1
#         # print("Prima: " + str(e_list))
#         for index in range(len(e_list) - 1, -1, -1):
#             # E --> I
#             if e_list[index][1] <= 0.5:
#                 duration_gamma = rg1.expovariate(gamma)
#                 i_list.append([e_list[index][0], duration_gamma])
#                 e_list.remove(e_list[index])
#             else:
#                 e_list[index][1] -= 1
#         for index in range(0, len(i_list)):
#             ngbs = graph.neighbors(i_list[index][0])
#             for ngb in ngbs:
#                 if ngb in s_list:
#                     r = rg1.uniform(0.0, 1.0)
#                     # S --> E
#                     if r < beta:
#                         duration_sigma = rg1.expovariate(sigma)
#                         e_list.append([ngb, duration_sigma])
#                         s_list.remove(ngb)

#
# def sim_SIR_old(graph, start_t, end_t):
#     global s_list
#     global i_list
#     global r_list
#     global s_t
#     global i_t
#     global r_t
#     global gamma
#     global beta
#     global step_p_day
#
#     gamma = gamma * (1 / step_p_day)
#     beta = beta * (1 / step_p_day)
#
#     for step in range(start_t, end_t):
#         # if (step % step_p_day) == 0:
#         s_t.append(len(s_list))
#         e_t.append(len(e_list))
#         i_t.append(len(i_list))
#         r_t.append(len(r_list))
#         # I --> R
#         for index in range(len(i_list) - 1, -1, -1):
#             # if infect[0] in part:
#             if i_list[index][1] <= 0.5:  # abbiamo superato la durata dell'infezione generata
#                 r_list.append(i_list[index][0])
#                 i_list.remove(i_list[index])
#             else:
#                 i_list[index][1] -= 1
#
#         for index in range(0, len(i_list)):
#             ngbs = graph.neighbors(i_list[index][0])
#             for ngb in ngbs:
#                 if ngb in s_list:
#                     r = rg1.uniform(0.0, 1.0)
#                     if r < beta:  # CONTAGIO
#                         duration_gamma = rg1.expovariate(gamma)
#                         i_list.append([ngb, duration_gamma])
#                         s_list.remove(ngb)


def print_tracing_count():
    print("\nCount tracing SEIR: ")
    print("s_t: " + str(s_t))
    print("e_t: " + str(e_t))
    print("i_t: " + str(i_t))
    print("r_t: " + str(r_t))
    print("is_t: " + str(is_t))
    print("qs_t: " + str(qs_t))
    print("qei_t: " + str(qei_t))


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


def print_contact_matrix():
    indx = 0
    for elem in contact_matrix:
        print("Nodo " + str(indx) + ": " + str(elem))
        indx += 1


def print_contact_list():
    index = 0
    for elem in contact_list:
        print("Nodo " + str(index) + ": " + str(elem))
        index += 1


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


def plot_tracing_result(filename, offset=0):
    time = []
    for i in range(0, len(s_t)):
        time.append(i / step_p_day)
    # time = time[offset:]
    # list_time_new = np.linspace(min(time), max(time), 1000)
    plt.plot(time, s_t, color='blue')
    plt.plot(time, e_t, color='orange')
    plt.plot(time, i_t, color='red')
    plt.plot(time, r_t, color='yellow')
    plt.plot(time, is_t, color='violet')
    plt.plot(time, qs_t, color='green')
    plt.plot(time, qei_t, color='purple')
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
    green_patch = mpatches.Patch(color='green', label='Quarantined S')
    purple_patch = mpatches.Patch(color='purple', label='Quarantined EI')
    violetpatch = mpatches.Patch(color='violet', label='Isolated')
    plt.legend(handles=[blue_patch, orange_patch, red_patch, yellow_patch, green_patch, purple_patch, violetpatch])

    plt.savefig("img/" + str(filename) + "_SEIR.png")
    plt.close()


def plot_seir_tracing(filename, offset=0):
    time = []
    offset = offset * step_p_day
    for i in range(0, len(s_t)):
        time.append(i / step_p_day)
    time = time[offset:]
    # list_time_new = np.linspace(min(time), max(time), 1000)
    plt.plot(time, qs_t[offset:], color='green')
    plt.plot(time, qei_t[offset:], color='violet')
    plt.plot(time, i_t[offset:], color='red')
    plt.plot(time, is_t[offset:], color='purple')
    # x_int = range(min(time), math.ceil(max(time)) + 1)
    # plt.xticks(x_int) # per avere interi nelle ascisse
    plt.title('Simulation Result - SEIR', fontsize=14)
    plt.xlabel('Time (gg)', fontsize=14, )
    plt.ylabel('', fontsize=14)
    plt.grid(True)

    # legend
    blue_patch = mpatches.Patch(color='green', label='Q-S')
    orange_patch = mpatches.Patch(color='violet', label='Q-IE')
    red_patch = mpatches.Patch(color='red', label='Infected')
    yellow_patch = mpatches.Patch(color='purple', label='Isolated')
    plt.legend(handles=[blue_patch, orange_patch, red_patch, yellow_patch])

    plt.savefig("img/" + str(filename) + "_SEIR.png")
    plt.close()


def plot_seir_result(filename, offset=0):
    time = []
    offset = offset * step_p_day
    for i in range(0, len(s_t)):
        time.append(i / step_p_day)
    time = time[offset:]
    # list_time_new = np.linspace(min(time), max(time), 1000)
    plt.plot(time, s_t[offset:], color='blue')
    plt.plot(time, e_t[offset:], color='orange')
    plt.plot(time, i_t[offset:], color='red')
    plt.plot(time, r_t[offset:], color='yellow')
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


# def epidemic(graph, n_days):
#     global beta
#     mu = gamma
#
#     # scelgo random gli infetti iniziali
#     list_nodes = list(graph.nodes())
#     rg1.shuffle(list_nodes)
#     initial_infections = list_nodes[0:n_inf]
#     sim = eon.fast_SIR(graph, beta, mu, initial_infecteds=initial_infections, tmax=n_days, return_full_data=True)
#     t = sim.t()
#     S = sim.S()  # numero suscettibili ad ogni istante
#     I = sim.I()  # numero infetti ad ogni istante
#     R = sim.R()  # numero rimossi ad ogni istante
#
#     r_per = R[-1] / len(graph.nodes()) * 100
#     s_per = S[-1] / len(graph.nodes()) * 100
#     i_per = I[-1] / len(graph.nodes()) * 100
#
#     # Print Result
#     plt.plot(t, S, label='S')
#     plt.plot(t, I, label='I')
#     plt.plot(t, R, label='R')
#     plt.legend()
#     plt.savefig('img/comparison_EON_beta_' + str(beta) + ' ' + 'mu_' + str(mu) + '_SIR.png')
#     plt.close()
#
#     # print('animation...')
#     # ani = sim.animate(ts_plots=['I', 'SIR'], node_size=4)
#     # writer = animation.PillowWriter('fps=2')
#     # # ani.save('SIR.mp4',writer=Writer,fps=5, extra_args=['-vcodec', 'libx264'])
#     # ani.save('compare_EON_beta_' + str(beta) + ' ' + 'mu_' + str(mu) + ' SIR.gif')
#     plt.close()


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
def get_tracing_result():
    global s_t
    global e_t
    global i_t
    global r_t
    global is_t
    global qs_t
    global qei_t
    print("Start avg calc ...")
    if abs:
        [s_t, e_t, i_t, r_t, is_t, qs_t, qei_t] = fm.calculate_average_from_csv_tracing(n)
    else:
        [s_t, e_t, i_t, r_t, is_t, qs_t, qei_t] = fm.calculate_average_from_csv_tracing()

    fm.write_csv_tracing(s_t, e_t, i_t, r_t, is_t, qs_t, qei_t, avg=True)
    print("End avg calc")
    plot_seir_result("avg_seir_n_sim=" + str(n_s * n_proc), offset=start_contagion)
    plot_seir_tracing("avg_tracing_n_sim=" + str(n_s * n_proc), offset=start_contagion)


def get_seir_result():
    global s_t
    global e_t
    global i_t
    global r_t

    print("Start avg calc ...")
    if abs:
        [s_t, e_t, i_t, r_t] = fm.calculate_average_from_csv_seir()
    else:
        [s_t, e_t, i_t, r_t] = fm.calculate_average_from_csv_seir()
    fm.write_csv_seir(s_t, e_t, i_t, r_t, avg=True)
    print("End avg calc")
    plot_seir_result("avg_seir_n_sim=" + str(n_s*n_proc), offset=start_contagion)


def plot_result_from_avg():
    global s_t
    global e_t
    global i_t
    global r_t
    global is_t
    global qs_t
    global qei_t
    [s_t, e_t, i_t, r_t, is_t, qs_t, qei_t] = fm.read_csv(tracing=True, avg=True)
    plot_seir_result("avg_seir", offset=start_contagion)
    plot_seir_tracing("avg_tracing=", offset=start_contagion)


def plot_avg_result():
    global s_t
    global e_t
    global i_t
    global r_t
    global is_t
    global qs_t
    global qei_t
    print("Start avg calc ...")
    [s_t, e_t, i_t, r_t, is_t, qs_t, qei_t] = fm.calculate_average_from_csv_tracing()
    fm.write_csv_tracing(s_t, e_t, i_t, r_t, is_t, qs_t, qei_t, avg=True)
    print("End avg calc")
    plot_seir_result("avg_seir_n_sim=" + str(n_s), offset=start_contagion)
    plot_seir_tracing("avg_tracing_n_sim=" + str(n_s), offset=start_contagion)


def parse_input_file():
    global beta
    global sigma
    global gamma
    global eta

    global start_contagion
    global window_size
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
    global n_proc

    global pr_diagnosis
    global pr_notification

    global is_sparse
    global fr_far_contacts

    global school_density
    global office_density
    global transport_density

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
        window_size = data["window_size"]
        is_sparse = data["is_sparse"] == True
        fr_far_contacts = data["fr_far_contacts"]
        start_contagion = data["start_contagion"]
        school_density = data["school_density"]
        office_density = data["office_density"]
        transport_density = data["transport_density"]
        n_proc = data["n_proc"]
        if data["Transport"] != 0:
            graph_ext.append("Transport")
        if data["Office"] != 0:
            graph_ext.append("Office")
        if data["School"] != 0:
            graph_ext.append("School")
        if data["Families"] != 0:
            graph_ext.append("Families")

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
        eta = data["eta"]

        print("\nEpidemic Parameters: \n")
        print("Beta: ..................... " + str(beta))
        print("Sigma: .................... " + str(sigma))
        print("Gamma: .................... " + str(gamma))
        print("Eta: ...................... " + str(eta))
        print("Number Initial Infected: .. " + str(n_inf))
        print()


def identify_process():
    global beta
    print("Start Process")
    beta = 1
    beta = beta + 1
    time.sleep(2)
    beta = beta + 1
    print(multiprocessing.current_process().name)
    print(beta)


def wrong_param():
    print("Wrong Paramters!")
    print("If you want to run seir simulation:\n> python epidemic_sim.py \"seir\"  n_s abs")
    print("  n_s: number of simulation")
    print("  abs: True --> graphic y are percantage, False --> y range is number of nodes")
    print("EXAMPLE: python epidemic_sim.py \"seir\" 10 True")
    print(
        "\nIf you want to run tracing simulation:\n> python epidemic_sim.py \"tracing\" n_s abs")
    print("  n_s: number of simulation")
    print("  abs: True --> graphic y are percantage, False --> y range is number of nodes")
    print("EXAMPLE: python epidemic_sim.py \"tracing\" 10 True")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "validation_1":
            print(sys.argv[1])
            first_validation(sys.argv[1])
        elif sys.argv[1] == "validation_2":
            print(sys.argv[1])
            first_validation(sys.argv[1])
        elif sys.argv[1] == "validation_3":
            print(sys.argv[1])
            second_validation(sys.argv[1])
        elif sys.argv[1] == "tracing_old":
            parse_input_file()
            simulate_tracing()
        elif sys.argv[1] == "result_from_avg":
            parse_input_file()
            plot_result_from_avg()
        elif sys.argv[1] == "result":
            parse_input_file()
            get_tracing_result()
        elif sys.argv[1] == "seir":
            n_s = int(sys.argv[2])
            if len(sys.argv) > 3:
                abs = bool(sys.argv[3])
            fm.clear_csv()
            fm.clear_avg_csv()
            parse_input_file()
            print("Numero simulazioni: " + str(n_s))
            run_sim(False)
            flush_structures()
            get_seir_result()
        elif sys.argv[1] == "tracing":
            n_s = int(sys.argv[2])
            if len(sys.argv) > 3:
                abs = bool(sys.argv[3])
            fm.clear_csv()
            fm.clear_avg_csv()
            parse_input_file()
            print("Numero Simulazioni: " + str(n_s))
            run_sim(True)
            flush_structures()
            get_tracing_result()

            # plot_seir_result()
            print("done")
            # parse_input_file()
            # simulate_tracing()
        elif sys.argv[1] == "multiproc":
            if len(sys.argv) < 3 or (sys.argv[2] != "tracing" and sys.argv[2] != "seir"):
                sys.exit("Second Parameter must be \"tracing\" or \"seir\"")
            tracing = sys.argv[2] == "tracing"
            if len(sys.argv) < 4:
                sys.exit("Insert number of simulation!")
            n_s = int(sys.argv[3])
            if len(sys.argv) < 5:
                sys.exit("Insert True if you want plot y as a probability!")
            abs = True == sys.argv[4]
            if n_proc > len(initial_seeds):
                sys.exit("n_proc must be leq then " + str(len(initial_seeds)) + "!")

            var = 2
            fm.clear_csv()
            fm.clear_avg_csv()
            parse_input_file()
            proc_list = []  # lista processi
            lock = Lock()
            for index in range(0, n_proc):
                proc_list.append(
                    Process(name=initial_seeds[index], args=(lock, initial_seeds[index]), target=proc_run_sim))
                proc_list[index].start()
            for proc in proc_list:
                proc.join()
                print("Process " + str(proc.name) + " has finished")
            print("Results processing...")
            flush_structures()
            if tracing:
                get_tracing_result()
            else:
                get_seir_result()
            print("You can download the results")
        elif sys.argv[1] == "write_res":
            # fm.clear_csv()
            # fm.clear_avg_csv()
            # s_t = [3, 5, 8]
            # e_t = [1, 1, 1]
            # i_t = [9, 9, 9]
            # r_t = [10, 1, 1]
            # fm.write_csv_seir(s_t, e_t, i_t, r_t)
            # s_t = [4, 5, 9]
            # e_t = [1, 1, 1]
            # i_t = [3, 5, 8]
            # r_t = [10, 1, 1]
            # fm.write_csv_seir(s_t, e_t, i_t, r_t)
            # s_t = [5, 5, 8]
            # e_t = [1, 5, 1]
            # i_t = [9, 9, 9]
            # r_t = [10, 1, 1]
            # fm.write_csv_seir(s_t, e_t, i_t, r_t)
            # [a, b, c, d] = fm.calculate_average_from_csv_seir()

            fm.clear_csv()
            s_t = [3, 5, 8]
            e_t = [1, 1, 1]
            i_t = [9, 9, 9]
            r_t = [10, 1, 1]
            is_t = [10, 1, 1]
            q_t = [10, 1, 1]
            fm.write_csv_tracing(s_t, e_t, i_t, r_t, is_t, q_t, q_t)
            s_t = [4, 5, 9]
            e_t = [1, 1, 1]
            i_t = [3, 5, 8]
            r_t = [10, 1, 1]
            is_t = [10, 1, 1]
            q_t = [10, 1, 1]
            fm.write_csv_tracing(s_t, e_t, i_t, r_t, is_t, q_t, q_t)
            s_t = [5, 5, 8]
            e_t = [1, 5, 1]
            i_t = [9, 9, 9]
            r_t = [10, 1, 1]
            is_t = [10, 2, 1]
            q_t = [10, 10, 1]
            fm.write_csv_tracing(s_t, e_t, i_t, r_t, is_t, q_t, q_t)
            [a, b, c, d, e, f, g] = fm.calculate_average_from_csv_tracing()
            print(a)
            print(b)
            print(c)
            print(d)
            print(e)
            print(f)
            print(g)
            fm.write_csv_tracing(a, b, c, d, e, f, g, True)

        elif sys.argv[1] == "seir":
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
            update_contacts(graph, 1)
            graph = gm.nx.erdos_renyi_graph(15, 0.1)
            update_contacts(graph, 2)
            print_contact_list()
        else:
            wrong_param()
    else:
        wrong_param()
