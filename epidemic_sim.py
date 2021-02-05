import graph_manager as gm
import random
import EoN as eon  # not used
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import numpy
from itertools import islice
from random import randint
import time
import yaml
import sys
import gc
import file_manager as fm

n = 0  # number of total node
home_step = 0  # step for days
work_step = 0  # step for days
n_days = 0  # number of days
beta = 0  # probability of contagion
sigma = 0  # transition rate from E to I
gamma = 0  # transition rate from I to R
fr_inf = 0  # number of initial infected
step_p_day = 0  # number of step per day
fr_station_user = 0  # fraction of people that use train station

s_list = []  # list of susceptible nodes
e_list = []  # list of exposed nodes
i_list = []  # list of infected nodes
r_list = []  # list of recovered/isolated/dead nodes

s_t = []  # number of susceptible for each step (e. g. step_p_day = 10 -> step = 1 / 10)
e_t = []  # number of exposed for each step
i_t = []  # number of infected for each step
r_t = []  # number of recovered/isolated/dead for each step

people_tot = []  # array of nodes
families = []  # list of families, each family is a list of nodes
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


def create_partions():
    global people_tot
    global fr_station_user

    random.shuffle(people_tot)
    n_station_user = int(fr_station_user * n)
    station_partitions = list(generate_partitions(people_tot[:n_station_user], 20, 100))
    public_transport_partitions = list(generate_partitions(people_tot[n_station_user:], 5, 40))
    return [station_partitions, public_transport_partitions]


def validation_0():
    global n_days
    global clock
    global n
    global beta
    global gamma
    global sigma
    global fr_inf
    global s_list
    global e_list
    global i_list
    global r_list
    global step_p_day
    global people_tot

    n = 3000
    n_days = 50
    beta = 0.02
    sigma = 0.2
    gamma = 0.1
    fr_inf = 0.02
    step_p_day = 1
    initialize()
    graph = gm.nx.erdos_renyi_graph(n, 0.05)
    initialize_Infected()
    interval_tot = n_days * step_p_day
    sim_SEIR(graph, 0, interval_tot)
    count_SEIR()


# def validation_1():
#     global n_days
#     global clock
#     global n
#     global beta
#     global gamma
#     global sigma
#     global people_tot
#     global families
#     global s_list
#     global e_list
#     global i_list
#     global r_list
#     global commuter_partitions
#     global fr_inf
#     global step_p_day
#     global s_t
#     global e_t
#     global i_t
#     global r_t
#
#     n = 500
#     n_days = 50
#     beta = 0.2
#     sigma = 0.05
#     gamma = 0.01
#     fr_inf = 50
#     step_p_day = 1
#
#     people_tot = [elem for elem in range(0, n * 4)]
#     part1 = people_tot[:n]
#     part2 = people_tot[n:2 * n]
#     part3 = people_tot[2 * n:3 * n]
#     part4 = people_tot[3 * n:4 * n]
#     initial_i = []
#     # print(part1)
#     # print(part2)
#     # input()
#     for i in range(0, n * 4, 10):
#         initial_i.append(i)
#     i_list = [[elem, 0] for elem in initial_i]
#     for elem in people_tot:
#         if is_infected(elem) == -1:
#             s_list.append(elem)
#     e_list = []
#     r_list = []
#     graph1 = gm.create_station_graph(part1, 0.1)
#     print("Part1")
#     print(part1)
#     graph2 = gm.create_station_graph(part2, 0.1)
#     print("Part2")
#     print(part2)
#     graph3 = gm.create_station_graph(part3, 0.1)
#     print("Part3")
#     print(part3)
#     graph4 = gm.create_station_graph(part4, 0.1)
#     print("Part4")
#     print(part4)
#     graph5 = gm.nx.union(graph1, graph2)
#     graph6 = gm.nx.union(graph5, graph3)
#     graph7 = gm.nx.union(graph6, graph4)
#     sim_SEIR(graph7, 0, n_days)
#     print_SEIR_count()
#     print_list()


def first_validation(validation_type):
    global n_days
    global clock
    global n
    global beta
    global gamma
    global sigma
    global fr_inf
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
    graph = gm.nx.erdos_renyi_graph(n, 0.05)
    flush_structures()
    initialize()
    initialize_Infected()
    interval_tot = n_days * step_p_day
    sim_SIR(graph, 0, interval_tot)
    print_SIR_count()
    plot_SIR_result(validation_type)
    flush_structures()
    initialize()
    initialize_Infected()
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
    global fr_inf
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

    graph = gm.nx.erdos_renyi_graph(n, 0.05)
    flush_structures()
    initialize()
    initialize_Infected()
    interval_tot = n_days * step_p_day
    sim_SIR(graph, 0, interval_tot)
    print_SIR_count()
    plot_SIR_result(graphic_name + "_" + str(step_p_day) + "_steps")
    flush_structures()
    initialize()
    step_p_day = 1
    initialize_Infected()
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


def initialize():
    global clock
    global fr_inf
    global step_p_day

    global people_tot
    global families
    global commuter_partitions
    global public_transport_users

    global s_list
    global e_list
    global i_list
    global r_list

    global s_t
    global e_t
    global i_t
    global r_t
    global fr_station_user

    random.seed(a=None)
    clock = 0
    # step_p_day = 1  # work_step + home_step
    people_tot = [elm for elm in range(0, n)]
    random.shuffle(people_tot)
    n_inf = int(fr_inf * n)
    initial_i = [el for el in people_tot[:n_inf]]
    i_list = [[el, 0] for el in initial_i]
    for elm in people_tot:
        if is_infected(elm) == -1:
            s_list.append(elm)
    e_list = []
    r_list = []


def initialize_Infected():
    for elem in i_list:
        elem[1] = random.expovariate(gamma)


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

    initialize_Infected()
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
        if (step % step_p_day) == 0:
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
                    if r < beta1:  # l'infetto contatta il vicino
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
        if (step % step_p_day) == 0:
            s_t.append(len(s_list))
            e_t.append(len(e_list))
            i_t.append(len(i_list))
            r_t.append(len(r_list))
        # I --> R
        for index in range(len(i_list) - 1, -1, -1):
            # if infect[0] in part:
            if i_list[index][1] <= 0.5:
                r_list.append(i_list[index][0])
                i_list.remove(i_list[index])
            else:
                i_list[index][1] -= 1
        # E --> I
        # print("Prima: " + str(e_list))
        for index in range(len(e_list) - 1, -1, -1):
            if e_list[index][1] <= 0.5:
                duration_gamma = random.expovariate(gamma1)
                i_list.append([e_list[index][0], duration_gamma])
                e_list.remove(e_list[index])
            else:
                e_list[index][1] -= 1
        # print("Dopo: " + str(e_list))
        for index in range(0, len(i_list)):
            ngbs = graph.neighbors(i_list[index][0])
            for ngb in ngbs:
                if ngb in s_list:
                    r = random.uniform(0.0, 1.0)
                    if r < beta1:  # l'infetto contatta il vicino
                        duration_sigma = random.expovariate(sigma1)
                        e_list.append([ngb, duration_sigma])
                        s_list.remove(ngb)


def sim_SEIRLessPrec(graph, start_t, end_t):
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
        if (step % step_p_day) == 0:
            s_t.append(len(s_list))
            e_t.append(len(e_list))
            i_t.append(len(i_list))
            r_t.append(len(r_list))
        # I --> R
        for index in range(len(i_list) - 1, -1, -1):
            # if infect[0] in part:
            if i_list[index][1] <= 0:  # abbiamo superato la durata dell'infezione generata
                r_list.append(i_list[index][0])
                i_list.remove(i_list[index])
            else:
                i_list[index][1] -= 1
        # E --> I
        # print("Prima: " + str(e_list))
        for index in range(len(e_list) - 1, -1, -1):
            if e_list[index][1] <= 0:
                duration_gamma = random.expovariate(gamma1)
                i_list.append([e_list[index][0], duration_gamma])
                e_list.remove(e_list[index])
            else:
                e_list[index][1] -= 1
        # print("Dopo: " + str(e_list))
        for index in range(0, len(i_list)):
            ngbs = graph.neighbors(i_list[index][0])
            for ngb in ngbs:
                if ngb in s_list:
                    r = random.uniform(0.0, 1.0)
                    if r < beta1:  # l'infetto contatta il vicino
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
        time.append(i + 1)
    # list_time_new = np.linspace(min(time), max(time), 1000)

    plt.plot(time, s_t, color='blue')
    plt.plot(time, i_t, color='red')
    plt.plot(time, r_t, color='yellow')
    plt.title('Simulation Result', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('', fontsize=14)
    plt.grid(True)

    # legend
    blue_patch = mpatches.Patch(color='blue', label='Susceptible')
    orange_patch = mpatches.Patch(color='orange', label='Exposed')
    red_patch = mpatches.Patch(color='red', label='Infected')
    yellow_patch = mpatches.Patch(color='yellow', label='Reduced')
    plt.legend(handles=[blue_patch, orange_patch, red_patch, yellow_patch])

    plt.savefig("img/" + str(filename) + "_SIR.png")
    plt.close()


def plot_SEIR_result(filename):
    time = []
    for i in range(0, len(s_t)):
        time.append(i + 1)
    # list_time_new = np.linspace(min(time), max(time), 1000)
    plt.plot(time, s_t, color='blue')
    plt.plot(time, e_t, color='orange')
    plt.plot(time, i_t, color='red')
    plt.plot(time, r_t, color='yellow')
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


def epidemic(graph, fr_inf):
    mu = 0.002
    beta = 0.3

    # scelgo random gli infetti iniziali
    list_nodes = list(graph.nodes())
    random.shuffle(list_nodes)
    initial_infections = list_nodes[0:fr_inf]

    print('Start SIR...')
    sim = eon.fast_SIR(graph, beta, mu, initial_infecteds=initial_infections, tmax=40, return_full_data=True)
    t = sim.t()
    S = sim.S()  # numero suscettibili ad ogni istante
    I = sim.I()  # numero infetti ad ogni istante
    R = sim.R()  # numero rimossi ad ogni istante

    r_per = R[-1] / len(graph.nodes()) * 100
    s_per = S[-1] / len(graph.nodes()) * 100
    i_per = I[-1] / len(graph.nodes()) * 100

    # Print Result

    print('S: ' + str(s_per) + '%\n' + 'I: ' + str(i_per) + '%\n' + 'R: ' + str(r_per) + '%\n')
    # plt.plot(t, S, label='S')
    # plt.plot(t, I, label='I')
    # plt.plot(t, R, label='R')
    # plt.legend()
    # plt.savefig('beta_' + str(beta) + ' ' + 'mu_' + str(mu) + '_SIR.png')
    # plt.close()

    print('done\n')
    print('animation...')
    ani = sim.animate(ts_plots=['I', 'SIR'], node_size=4)
    # writer = animation.PillowWriter('fps=2')
    # ani.save('SIR.mp4',writer=Writer,fps=5, extra_args=['-vcodec', 'libx264'])
    ani.save('beta_' + str(beta) + ' ' + 'mu_' + str(mu) + ' SIR.gif')
    plt.close()
    print('done\n')


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
    global n
    global n_days
    global beta
    global sigma
    global gamma
    global fr_inf
    global fr_station_user
    global step_p_day

    with open("config_files/graph_and_time_parameters.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        n = data["n_nodes"]  # number of total node
        n_days = data["n_days"]  # number of days
        fr_station_user = data["fr_station_user"]
        step_p_day = data["step_p_day"]

        print("\nGraph and Time Parameters: \n")
        print("Number of Nodes: .......... " + str(n))
        print("n days: ................... " + str(n_days))
        print("Perc Station User.......... " + str(fr_station_user))
        print("Step per Day............... " + str(step_p_day))
        print()

    with open("config_files/epidemic_parameters.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        beta = data["beta"]  # probability of contagion
        sigma = data["sigma"]  # transition rate from E to I
        gamma = data["gamma"]  # transition rate from I to R
        fr_inf = data["fr_inf"]  # number of initial infected

        print("\nEpidemic Parameters: \n")
        print("Beta: ..................... " + str(beta))
        print("Sigma: .................... " + str(sigma))
        print("Gamma: .................... " + str(gamma))
        print("Perc Infected: ............ " + str(fr_inf))
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
        G1 = gm.create_school_graph([elem for elem in range(80, 130)], 0.2)

        gm.print_graph_with_labels_and_neighb(G1)
    else:
        parse_input_file()
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
