import graph_manager as gm
import random
import EoN as eon  # not used
import matplotlib.pyplot as plt
from itertools import islice
from random import randint
import time
import matplotlib.patches as mpatches
import yaml
import sys

n = 0  # number of total node
home_step = 0  # step for days
work_step = 0  # step for days
n_days = 0  # number of days
beta = 0  # probability of contagion
sigma = 0  # transition rate from E to I
gamma = 0  # transition rate from I to R
n_inf = 0  # number of initial infected
step_p_day = 0  # number of step per day

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
station_users = []  # list of list of people that use one specific station/bus
clock = 0


def generate_partitions(input_list, min_size=1, max_size=6):
    it = iter(input_list)
    while True:
        nxt = list(islice(it, randint(min_size, max_size)))
        if nxt:
            yield nxt
        else:
            break


def count_SEIR():
    global s_t
    global e_t
    global i_t
    global r_t
    s_t.append(len(s_list))
    e_t.append(len(e_list))
    i_t.append(len(i_list))
    r_t.append(len(r_list))


def validation():
    global n_days
    global clock
    global n
    global beta
    global gamma
    global sigma
    global n_inf

    n = 3000
    n_days = 20
    beta = 0.01
    sigma = 0.33
    gamma = 0.01
    n_inf = 50
    step_p_day = 1

    initialize()
    graph = gm.nx.erdos_renyi_graph(n, 0.1)
    part = list(graph.nodes)
    interval_tot = n_days * step_p_day
    sim_SEIR(graph, 0, interval_tot, part, True)
    count_SEIR()


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


def validation2():
    global n_days
    global clock
    global n
    global beta
    global gamma
    global sigma
    global people_tot
    global families
    global s_list
    global e_list
    global i_list
    global r_list
    global station_users
    global n_inf
    global step_p_day
    global s_t
    global e_t
    global i_t
    global r_t

    n = 500
    n_days = 20
    beta = 0.2
    sigma = 0.33
    gamma = 0.01
    n_inf = 50
    step_p_day = 1

    people_tot = [elem for elem in range(0, n * 4)]
    part1 = people_tot[:n]
    part2 = people_tot[n:2 * n]
    part3 = people_tot[2 * n:3 * n]
    part4 = people_tot[3 * n:4 * n]
    initial_i = []
    # print(part1)
    # print(part2)
    # input()
    for i in range(0, n * 4, 10):
        initial_i.append(i)
    i_list = [[elem, 0] for elem in initial_i]
    for elem in people_tot:
        if is_infected(elem) == -1:
            s_list.append(elem)
    e_list = []
    r_list = []
    # print_list()
    # input()
    start_t = 0
    end_t = 20
    graph = gm.create_station_graph(part1, 0.1)
    part = list(graph.nodes)
    # print_list()
    # input()
    print_list()
    count_SEIR()
    sim_SEIR(graph, start_t, end_t, part, False)
    print("Part1")
    print_list()
    print_SEIR_count()
    print(part)
    input()
    graph = gm.create_station_graph(part2, 0.1)
    part = list(graph.nodes)
    interval_tot = n_days * step_p_day
    sim_SEIR(graph, 0, interval_tot, part, False)
    print("Part2")
    print(graph.nodes)
    print_list()

    print_SEIR_count()
    print(part)
    input()
    graph = gm.create_station_graph(part3, 0.1)
    part = list(graph.nodes)
    sim_SEIR(graph, 0, interval_tot, part, False)
    print("Part3")
    print_list()

    print_SEIR_count()
    input()
    graph = graph = gm.create_station_graph(part4, 0.1)
    part = list(graph.nodes)
    sim_SEIR(graph, 0, interval_tot, part, True)
    print("Part4")
    print_list()
    count_SEIR()
    print_SEIR_count()
    input()


def validation3():
    global n_days
    global clock
    global n
    global beta
    global gamma
    global sigma
    global people_tot
    global families
    global s_list
    global e_list
    global i_list
    global r_list
    global station_users
    global n_inf
    global step_p_day
    global s_t
    global e_t
    global i_t
    global r_t

    n = 500
    n_days = 20
    beta = 0.2
    sigma = 0.33
    gamma = 0.01
    n_inf = 50
    step_p_day = 1

    people_tot = [elem for elem in range(0, n * 4)]
    part1 = people_tot[:n]
    part2 = people_tot[n:2 * n]
    part3 = people_tot[2 * n:3 * n]
    part4 = people_tot[3 * n:4 * n]
    initial_i = []
    # print(part1)
    # print(part2)
    # input()
    for i in range(0, n * 4, 10):
        initial_i.append(i)
    i_list = [[elem, 0] for elem in initial_i]
    for elem in people_tot:
        if is_infected(elem) == -1:
            s_list.append(elem)
    e_list = []
    r_list = []
    # print_list()
    # input()
    start_t = 0
    end_t = 20
    graph1 = gm.create_station_graph(part1, 0.1)
    print("Part1")
    print(part1)
    graph2 = gm.create_station_graph(part2, 0.1)
    print("Part2")
    print(part2)
    graph3 = gm.create_station_graph(part3, 0.1)
    print("Part3")
    print(part3)
    graph4 = gm.create_station_graph(part4, 0.1)
    print("Part4")
    print(part4)
    graph5 = gm.nx.union(graph1, graph2)
    graph6 = gm.nx.union(graph5, graph3)
    graph7 = gm.nx.union(graph6, graph4)
    sim_SEIR(graph7, 0, 20)
    print_list()
    input()


def initialize():
    global clock
    global people_tot
    global families
    global s_list
    global e_list
    global i_list
    global r_list
    global station_users
    global n_inf
    global step_p_day
    global s_t
    global e_t
    global i_t
    global r_t

    random.seed(a=None)
    clock = 0
    step_p_day = 1  # work_step + home_step
    people_tot = [elem for elem in range(0, n)]
    random.shuffle(people_tot)
    initial_i = [elem for elem in people_tot[:n_inf]]
    i_list = [[elem, 0] for elem in initial_i]
    for elem in people_tot:
        if is_infected(elem) == -1:
            s_list.append(elem)
    e_list = []
    r_list = []
    # print(sorted(s_list))
    # print(sorted(i_list))
    # print(sorted(people_tot))
    # input()
    random.shuffle(people_tot)
    families = list(generate_partitions(people_tot, 1, 6))
    random.shuffle(people_tot)
    station_users = list(generate_partitions(people_tot, 50, 400))

    for elem in i_list:
        elem[1] = random.expovariate(gamma)  # setto durata infezioni


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
    global station_user

    print("Starting Simulation...")
    start_time = time.time()

    for day in range(0, n_days):
        for elem in families:
            graph = gm.create_home_graph(elem)
            sim_SEIR(graph, clock, (clock + home_step), elem)
        clock += home_step

        for elem1 in station_users:
            graph = gm.create_station_graph(elem1)
            sim_SEIR(graph, clock, (clock + work_step), elem1)
        clock += work_step
        print(e_list)

    end_time = time.time()
    duration = round((end_time - start_time), 3)
    print("duration SEIR simulation: " + str(duration) + " Seconds")
    # time += n_days
    # print(s_t)
    # print(i_t)
    # print(r_t)


def sim_SEIR(graph, start_t, end_t, part=None, last_part=False):
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
        # if last_part:
        #     count_SEIR()
        s_t.append(len(s_list))
        e_t.append(len(e_list))
        i_t.append(len(i_list))
        r_t.append(len(r_list))
        # I --> R
        for infect in i_list:
            # if infect[0] in part:
            if infect[1] <= 0:  # abbiamo superato la durata dell'infezione generata
                r_list.append(infect[0])
                i_list.remove(infect)
            else:
                infect[1] -= 1
        # E --> I
        for exp in e_list:
            # if exp[0] in part:
            if exp[1] <= 0:  # abbiamo superato la durata dell'esposizione generata
                duration_gamma = random.expovariate(gamma1)
                i_list.append([exp[0], duration_gamma])
                e_list.remove(exp)
            else:
                exp[1] -= 1

        for elem in i_list:
            # if elem[0] in part:
            ngbs = graph.neighbors(elem[0])
            for ngb in ngbs:
                if ngb in s_list:
                    r = random.uniform(0.0, 1.0)
                    if r < beta1:  # l'infetto contatta il vicino
                        duration_sigma = random.expovariate(sigma)
                        e_list.append([ngb, duration_sigma])
                        s_list.remove(ngb)

    return [s_t, i_t, r_t]


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


def plot_SIR_result():
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
    plt.savefig("result.png")
    plt.close()


def plot_SEIR_result():
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
    plt.savefig("img/result_SEIR.png")
    plt.close()


def epidemic(graph, n_inf):
    mu = 0.002
    beta = 0.3

    # scelgo random gli infetti iniziali
    list_nodes = list(graph.nodes())
    random.shuffle(list_nodes)
    initial_infections = list_nodes[0:n_inf]

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
    global home_step
    global work_step
    global n_days
    global beta
    global sigma
    global gamma
    global n_inf

    with open("config_files/input_parameters.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        n = data["n_nodes"]  # number of total node
        home_step = data["home_step"]  # step for days
        work_step = data["work_step"]  # step for days
        n_days = data["n_days"]  # number of days
        beta = data["beta"]  # probability of contagion
        sigma = data["sigma"]  # transition rate from E to I
        gamma = data["gamma"]  # transition rate from I to R
        n_inf = data["n_inf"]  # number of initial infected

        print("\nInput Parameters: \n")
        print("number of nodes: .......... " + str(n))
        print("home step: ................ " + str(home_step))
        print("work step: ................ " + str(work_step))
        print("n days: ................... " + str(n_days))
        print("beta: ..................... " + str(beta))
        print("sigma: .................... " + str(sigma))
        print("gamma: .................... " + str(gamma))
        print("n infetti: ................ " + str(n_inf))
        print()


if __name__ == '__main__':

    if sys.argv[1] == "validation":
        print(sys.argv[1])
        validation()
        plot_SEIR_result()
    elif sys.argv[1] == "validation3":
        print(sys.argv[1])
        validation3()
        plot_SEIR_result()
    elif sys.argv[1] == "test":
        graph1 = gm.create_station_graph([1, 2, 3, 4])
        graph2 = gm.create_tube_graph([5, 6, 7, 8])
        # elem = graph1.nodes[1]["graph_name"]
        res = gm.nx.union(graph1, graph2)
        for elem in res:
            print(elem)
            print(res.nodes[elem]["graph_name"])
        #gm.write_graph(res, "grafo_label")
        gm.write_labeled_graph(res, "prova")
        res1 = gm.read_labeled_graph("test")
        print("\nGrafo letto da File...")
        for elem in res1:
            print(elem)
            print(res1.nodes[elem]["graph_name"])
    else:
        parse_input_file()
        simulate()
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
