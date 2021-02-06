import graph_manager as gm
import random
import EoN as eon  # not used
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from itertools import islice
from random import randint
import time
import yaml
import sys
import gc
import file_manager as fm

home_step = 0  # step for days
work_step = 0  # step for days
n_days = 0  # number of days
beta = 0  # probability of contagion
sigma = 0  # transition rate from E to I
gamma = 0  # transition rate from I to R
fr_inf = 0  # number of initial infected
step_p_day = 0  # number of step per day
fr_station_user = 0  # fraction of people that use train station

# COUNTS
n = 0  # number of total node
n_inf = 0  # number of initial infected
n_stud = 0  # number of student
n_employs = 0  # number of employees
min_family_size = 0
max_familiy_size = 0
min_school_size = 0
max_school_size = 0
min_office_size = 0
max_office_size = 0

s_list = []  # list of susceptible nodes
e_list = []  # list of exposed nodes
i_list = []  # list of infected nodes
r_list = []  # list of recovered/isolated/dead nodes

s_t = []  # number of susceptible for each step (e. g. step_p_day = 10 -> step = 1 / 10)
e_t = []  # number of exposed for each step
i_t = []  # number of infected for each step
r_t = []  # number of recovered/isolated/dead for each step

people_tot = []  # array of nodes  NON SERVE!
traced_people = []  # One entry for eah person: The values are 0 -> S, 1 -> Isolated, 2 -> in quarantine
families = []  # list of families, each family is a list of nodes
school_partitions = []  # list of students grouped by school
office_partitions = []  # list of employees grouped by office

commuter_partitions = []  # list of list of people that use one specific station
public_transport_users = []  # list of list of people that use one specific public_transport/bus
clock = 0
precision_ctrl = True  # if True the simulator choose if the node change state when his trace time is less than 1


def generate_partitions(input_list, min_size=1, max_size=6):
    it = iter(input_list)
    while True:
        nxt = list(islice(it, randint(min_size, max_size)))
        if nxt:
            yield nxt
        else:
            break


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
    flush_structures()
    initialize()
    initialize_infected()
    graph = gm.nx.erdos_renyi_graph(n, 0.2)
    #print("Start EON Simulation")
    epidemic(graph, n_days)
    # print("End EON Simulation")
    flush_structures()
    initialize()
    initialize_infected()
    print("Start sim_SIR Simulation")
    sim_SIR(graph, 0, n_days * step_p_day)
    print("End sim_SIR Simulation")
    plot_SIR_result("comparison_sim")


def create_partions():
    global people_tot
    global fr_station_user

    random.shuffle(people_tot)
    n_station_user = int(fr_station_user * n)
    station_partitions = list(generate_partitions(people_tot[:n_station_user], 20, 100))
    public_transport_partitions = list(generate_partitions(people_tot[n_station_user:], 5, 40))
    return [station_partitions, public_transport_partitions]


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


def first_validation(validation_type):
    global n_days
    global clock
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

    with open("config_files/" + str(validation_type) + "_input.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        n = data["n_nodes"]  # number of total node
        n_days = data["n_days"]  # number of days
        n_days = data["n_days"]
        beta = data["beta"]
        sigma = data["sigma"]
        gamma = data["gamma"]
        fr_inf = data["fr_inf"]
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
    graph = gm.nx.erdos_renyi_graph(n, 0.05)
    flush_structures()
    initialize()
    initialize_infected()
    interval_tot = n_days * step_p_day
    sim_SIR(graph, 0, interval_tot)
    print_SIR_count()
    plot_SIR_result(validation_type)
    flush_structures()
    initialize()
    initialize_infected()
    sim_SEIR(graph, 0, interval_tot)
    print_SEIR_count()
    plot_SEIR_result(validation_type)


def second_validation(graphic_name):
    global n_days
    global clock
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
    graph = gm.nx.erdos_renyi_graph(n, 0.05)
    flush_structures()
    initialize()
    initialize_infected()
    interval_tot = n_days * step_p_day
    sim_SIR(graph, 0, interval_tot)
    print_SIR_count()
    plot_SIR_result(graphic_name + "_" + str(step_p_day) + "_steps")
    flush_structures()
    initialize()
    step_p_day = 1
    initialize_infected()
    interval_tot = n_days
    sim_SIR(graph, 0, interval_tot)
    print_SIR_count()
    plot_SIR_result(graphic_name + "_1_step")


def flush_structures():
    global s_list
    global e_list
    global i_list
    global r_list

    global s_t
    global e_t
    global i_t
    global r_t

    s_list = []
    e_list = []
    i_list = []
    r_list = []

    s_t = []
    e_t = []
    i_t = []
    r_t = []

    gc.get_stats()


def create_families_network():
    global families
    random.shuffle(people_tot)
    random.shuffle(people_tot)
    families = list(generate_partitions(people_tot, min_family_size, max_familiy_size))
    print("families: " + str(families))


def create_school_work_network():
    global school_partitions
    global office_partitions
    random.shuffle(people_tot)
    office_partitions = list(generate_partitions(people_tot[:n_stud], min_office_size, max_office_size))
    school_partitions = list(
        generate_partitions(people_tot[n_stud:(n_stud + n_employs)], min_school_size, max_school_size))
    print("office partitions: " + str(office_partitions))
    print("school partitions: " + str(school_partitions))


def initialize():
    global clock
    global fr_inf
    global step_p_day
    global people_tot
    global n_inf

    global s_list
    global e_list
    global i_list
    global r_list

    random.seed(a=None)
    clock = 0
    people_tot = [elm for elm in range(0, n)]
    random.shuffle(people_tot)
    initial_i = [el for el in people_tot[:n_inf]]
    i_list = [[el, 0] for el in initial_i]
    for elm in people_tot:
        if is_infected(elm) == -1:
            s_list.append(elm)
    e_list = []
    r_list = []


def initialize_infected():
    for infect in i_list:
        infect[1] = random.expovariate(gamma)


def is_infected(elem):
    ctrl = True
    i = 0
    while ctrl and i < len(i_list):
        ctrl = not (elem == i_list[i][0])
        i += 1
    if ctrl:
        return -1
    return i - 1


def is_exposed(elem):
    ctrl = True
    i = 0
    while ctrl and i < len(e_list):
        ctrl = not (elem == e_list[i][0])
        i += 1
    if ctrl:
        return -1
    return i - 1


def simulate():
    initialize()
    initialize_infected()
    print("Infetti iniziali")
    print(i_list)
    # # SIMULAZIONE PRIMO CICLO
    # day = 100
    # create_families_network()
    # temp_graphs = []
    # for family in families:
    #     temp_graphs.append(gm.create_home_graph(family))
    # fam_gr = gm.nx.union_all(temp_graphs)
    # for index in range(0, 50, 5):
    #     sim_SEIR(fam_gr, 0, day)
    # print("Infetti finali")
    # for elem in i_list:
    #     print(elem[0])

    # SIMULAZIONE SECONDO CICLO
    create_school_work_network()
    temp_graphs = []
    for office in office_partitions:
        temp_graphs.append(gm.create_office_graph(office, 0.3))
    for school in school_partitions:
        temp_graphs.append(gm.create_school_graph(school, 0.3))
    office_school_tot = gm.nx.union_all(temp_graphs)
    sim_SEIR(office_school_tot, 0, 50)
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


def simulate_1():
    global n_days
    global clock
    global step_p_day
    global families

    station_partitions = []
    public_transport_partitions = []
    initialize()
    station_partitions, public_transport_partitions = create_partions()
    start_time = time.time()
    graph = gm.nx.Graph()
    for elm in station_partitions:
        temp = gm.create_station_graph(elm)
        graph = gm.nx.union(graph, temp)
    for elm in public_transport_partitions:
        temp = gm.create_public_transport_graph(elm)
        graph = gm.nx.union(graph, temp)

    initialize_infected()
    sim_SEIR(graph, 0, 100)
    plot_SEIR_result("simulation")
    # gm.print_graph_with_labels_and_neighb(graph)

    end_time = time.time()
    duration = round((end_time - start_time), 3)
    print("duration SEIR simulation: " + str(duration) + " Seconds")
    # time += n_days
    # print(s_t)
    # print(i_t)
    # print(r_t)


def sim_SIR(graph, start_t, end_t):
    global s_list
    global i_list
    global r_list
    global s_t
    global i_t
    global r_t
    global gamma
    global beta

    gamma1 = gamma * (1 / step_p_day)
    beta1 = beta * (1 / step_p_day)

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
                    r = random.uniform(0.0, 1.0)
                    if r < beta1:  # CONTAGIO
                        duration_gamma = random.expovariate(gamma1)
                        i_list.append([ngb, duration_gamma])
                        s_list.remove(ngb)


def sim_SEIR(graph, start_t, end_t):
    global s_list
    global e_list
    global i_list
    global r_list
    global s_t
    global e_t
    global i_t
    global r_t

    gamma1 = gamma * (1 / step_p_day)
    sigma1 = sigma * (1 / step_p_day)
    beta1 = beta * (1 / step_p_day)

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
                duration_gamma = random.expovariate(gamma1)
                i_list.append([e_list[index][0], duration_gamma])
                e_list.remove(e_list[index])
            else:
                e_list[index][1] -= 1
        for index in range(0, len(i_list)):
            ngbs = graph.neighbors(i_list[index][0])
            for ngb in ngbs:
                if ngb in s_list:
                    r = random.uniform(0.0, 1.0)
                    # S --> E
                    if r < beta1:
                        duration_sigma = random.expovariate(sigma1)
                        e_list.append([ngb, duration_sigma])
                        s_list.remove(ngb)


def sim_SEIRIneff(graph, start_t, end_t, part):
    global s_list
    global i_list
    global r_list
    global s_t
    global e_t
    global i_t
    global r_t

    gamma1 = gamma * (1 / step_p_day)
    sigma1 = sigma * (1 / step_p_day)
    beta1 = beta * (1 / step_p_day)

    for step in range(start_t, end_t):
        # if (step % step_p_day) == 0:
        # print(i_list)
        s_t.append(len(s_list))
        e_t.append(len(e_list))
        i_t.append(len(i_list))
        r_t.append(len(r_list))

        # I --> R
        for infect in i_list:
            if infect[0] in part:
                if infect[1] <= 0:  # abbiamo superato la durata dell'infezione generata
                    r_list.append(infect[0])
                    i_list.remove(infect)
                else:
                    infect[1] -= 1
        # E --> I
        for exp in e_list:
            if exp[0] in part:
                if exp[1] <= 0:  # abbiamo superato la durata dell'esposizione generata
                    duration_gamma = random.expovariate(gamma1)
                    i_list.append([exp[0], duration_gamma])
                    e_list.remove(exp)
                else:
                    exp[1] -= 1

        for elem in i_list:
            if elem[0] in part:
                ngbs = graph.neighbors(elem[0])
                for ngb in ngbs:
                    if ngb in s_list:
                        r = random.uniform(0.0, 1.0)
                        if r < beta1:  # l'infetto contatta il vicino
                            duration_sigma = random.expovariate(sigma)
                            e_list.append([ngb, duration_sigma])
                            s_list.remove(ngb)

    return [s_t, i_t, r_t]


def print_SEIR_count():
    global s_t
    global e_t
    global i_t
    global r_t
    print("\nCount SEIR: ")
    print("s_t: " + str(s_t))
    print("e_t: " + str(e_t))
    print("i_t: " + str(i_t))
    print("r_t: " + str(r_t))


def print_SIR_count():
    global s_t
    global i_t
    global r_t
    print("\nCount SIR: ")
    print("s_t: " + str(s_t))
    print("i_t: " + str(i_t))
    print("r_t: " + str(r_t))


def print_list():
    global s_list
    global e_list
    global i_list
    global r_list
    print("\nLists: ")
    print("s_list: " + str(sorted(s_list)))
    print("e_list: " + str(e_list))
    print("i_list: " + str(i_list))
    print("r_list: " + str(sorted(r_list)))


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
        time.append(i/step_p_day)
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


def count_SEIR():
    global s_t
    global e_t
    global i_t
    global r_t
    s_t.append(len(s_list))
    e_t.append(len(e_list))
    i_t.append(len(i_list))
    r_t.append(len(r_list))


def epidemic(graph, n_days):
    global beta
    mu = gamma

    # scelgo random gli infetti iniziali
    list_nodes = list(graph.nodes())
    random.shuffle(list_nodes)
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


def change_grained():
    global i_t
    global s_t
    global r_t
    global e_t
    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []

    s_t = list(generate_partitions(s_t, step_p_day, step_p_day))
    e_t = list(generate_partitions(e_t, step_p_day, step_p_day))
    i_t = list(generate_partitions(i_t, step_p_day, step_p_day))
    r_t = list(generate_partitions(r_t, step_p_day, step_p_day))

    for index in range(0, len(s_t)):
        temp1.append(sum(s_t[index]))
        temp4.append(sum(e_t[index]))
        temp2.append(sum(i_t[index]))
        temp3.append(sum(r_t[index]))

    s_t = temp1
    e_t = temp4
    i_t = temp2
    r_t = temp3


def parse_input_file():
    global beta
    global sigma
    global gamma

    global n
    global n_inf
    global n_stud
    global n_employs

    global min_family_size
    global max_familiy_size
    global min_school_size
    global max_school_size
    global min_office_size
    global max_office_size

    global step_p_day
    global n_days

    with open("config_files/graph_and_time_parameters.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        n = data["n_nodes"]  # number of total node
        n_days = data["n_days"]  # number of days
        step_p_day = data["step_p_day"]
        n_stud = int(data["fr_students"] * n)
        n_employs = int(data["fr_employees"] * n)
        min_family_size = data["min_family_size"]
        max_familiy_size = data["max_familiy_size"]
        min_school_size = data["min_school_size"]
        max_school_size = data["max_school_size"]
        min_office_size = data["min_office_size"]
        max_office_size = data["max_office_size"]

        print("\nGraph and Time Parameters: \n")
        print("Number of Nodes: .......... " + str(n))
        print("n days: ................... " + str(n_days))
        print("Step per Day............... " + str(step_p_day))
        print("Number of Students: ....... " + str(n_stud))
        print("Number of Employee: ....... " + str(n_employs))
        print()
        print("Family size: .............. " + str(min_family_size) + " - " + str(max_familiy_size))
        print("School size: .............. " + str(min_school_size) + " - " + str(max_school_size))

    with open("config_files/epidemic_parameters.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        beta = data["beta"]  # probability of contagion
        sigma = data["sigma"]  # transition rate from E to I
        gamma = data["gamma"]  # transition rate from I to R
        n_inf = int(data["fr_inf"] * n)  # number of initial infected

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
    elif sys.argv[1] == "test":
        graph1 = gm.create_station_graph([1, 2, 3, 4])
        graph2 = gm.create_public_transport_graph([5, 6, 7, 8])
        # elem = graph1.nodes[1]["graph_name"]
        res = gm.nx.union(graph1, graph2)
        for elem in res:
            print(elem)
            print(res.nodes[elem]["graph_name"])
        # gm.write_graph(res, "grafo_label")
        gm.write_labeled_graph(res, "prova")
        res1 = gm.read_labeled_graph("test")
        print("\nGrafo letto da File...")
        for elem in res1:
            print(elem)
            print(res1.nodes[elem]["graph_name"])
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
    elif sys.argv[1] == "test_gr":
        # G1 = gm.create_station_graph([elem for elem in range(80, 130)], 0.2)
        # G1 = gm.create_public_transport_graph([elem for elem in range(80, 130)], 0.2)
        # G1 = gm.create_office_graph([elem for elem in range(80, 130)], 0.2)
        # G1 = gm.create_home_graph([8,2,3,4,5])
        # G1 = gm.create_school_graph([elem for elem in range(80, 2000)], 0.2)
        # G2 = gm.create_station_graph([elem for elem in range(2000, 4000)], 0.2)
        graphs = []
        for elem in range(0, 4000, 4):
            graphs.append(gm.create_home_graph([elem, elem + 1, elem + 2, elem + 3]))
        print("grafi creati")
        G3 = gm.nx.union_all(graphs)
        # gm.print_graph_with_labels_and_neighb(G3)
    elif sys.argv[1] == "simulate":
        parse_input_file()
        simulate()
    elif sys.argv[1] == "compare":
        compare_with_EON("comparison")
    else:
        parse_input_file()
        initialize()
        create_families_network()
        simulate()
        # simulate()
    # plot_SEIR_result()

    # g7 = gm.create_school_graph([4, 6, 8, 9, 13, 23, 45, 46, 47], 0.3)
    # print(gm.nx.edges(g7))
    # input()
    # simulate()
    # simulate()
    # validation()

    # print(people_tot)
    # print(families)

    # g2 = gm.create_station_graph(30)
    # gm.print_graph(g2, "station1")
    # gm.write_graph(g2, "station")
    # g3 = gm.read_graph("station")
    # gm.print_graph(g3, "station")
    # [s_t, i_t, r_t] = simulation_SIR(g3, 1)
    # plot_SIR_result(s_t, i_t, r_t)
    # epidemic(g2, 6)
