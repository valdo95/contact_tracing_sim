import graph_manager as gm
import random
import EoN as eon  # not used
import matplotlib.pyplot as plt
from itertools import islice
from random import randint
import time
import matplotlib.patches as mpatches
import yaml

n = 0  # number of total node
home_step = 0  # step for days
work_step = 0  # step for days
n_days = 0  # number of days
beta = 0  # probability of contagion
sigma = 0  # transition rate from E to I
gamma = 0  # transition rate from I to R
n_inf = 0  # number of initial infected

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
station_user = []  # list of list of people that use one specific station/bus
clock = 0


def generate_partitions(input_list, min_size=1, max_size=6):
    it = iter(input_list)
    while True:
        nxt = list(islice(it, randint(min_size, max_size)))
        if nxt:
            yield nxt
        else:
            break


def validation():
    global n_days
    global clock
    graph = gm.nx.erdos_renyi_graph(n, 0.1)
    # gm.write_graph(graph, "validation8000_graph")
    print("Start Reading AdjList...")
    # graph = gm.read_graph("validation8000_graph");
    part = list(graph.nodes)
    # graph = gm.read_graph("validation_graph")
    interval_tot = n_days * step_p_day
    sim_SEIR(graph, interval_tot, part)
    # change_grained()


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

    random.seed(a=None)
    clock = 0
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

    interval_tot = n_days * step_p_day
    print(families)
    for elem in families:
        print(elem)
        graph = gm.create_home_graph(elem)
        sim_SEIR(graph, interval_tot, elem)

    clock += interval_tot

    # n_days = 40
    # for elem in station_users:
    #     graph = gm.create_station_graph(elem)
    #     sim_SIR(graph, n_days)
    # time += n_days
    print(s_t)
    print(i_t)
    print(r_t)


def sim_SEIR(graph, interval_tot, part):
    print("Starting Simulation...")
    global s_list
    global i_list
    global r_list
    global s_t
    global i_t
    global r_t
    gamma1 = gamma * (1 / step_p_day)
    sigma1 = sigma * (1 / step_p_day)
    beta1 = beta * (1 / step_p_day)

    start_time = time.time()
    for elem in i_list:
        elem[1] = random.expovariate(gamma)  # setto durata infezioni

    for step in range(0, interval_tot):  # aggiorno conteggio una volta al giorno
        if (step % step_p_day) == 0:
            # print(i_list)
            s_t.append(len(s_list))
            e_t.append(len(e_list))
            i_t.append(len(i_list))
            r_t.append(len(r_list))

        # I --> R
        for infect in i_list:
            if infect[1] <= 0:  # abbiamo superato la durata dell'infezione generata
                r_list.append(infect[0])
                i_list.remove(infect)
            else:
                infect[1] -= 1
        # E --> I
        for exp in e_list:
            print(exp[0])
            print(part)
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
    end_time = time.time()
    duration = round((end_time - start_time), 3)
    print("duration SEIR simulation: " + str(duration) + " Seconds")
    return [s_t, i_t, r_t]


def sim_SEIROld(graph, interval_tot):
    global s_list
    global i_list
    global r_list
    global s_t
    global i_t
    global r_t
    # marked_i = {elem: False for elem in graph.nodes}
    gamma1 = gamma * (1 / step_p_day)
    sigma1 = sigma * (1 / step_p_day)
    beta1 = beta * (1 / step_p_day)
    for elem in i_list:
        elem[2] = random.expovariate(gamma)  # setto durata infezioni

    for step in range(0, interval_tot):  # aggiorno conteggio una volta al giorno
        if (step % step_p_day) == 0:
            print(i_list)
            s_t.append(len(s_list))
            e_t.append(len(e_list))
            i_t.append(len(i_list))
            r_t.append(len(r_list))

        # I --> R
        for infect in i_list:
            if infect[2] <= infect[1]:  # abbiamo superato la durata dell'infezione generata
                r_list.append(infect[0])
                i_list.remove(infect)
            else:
                infect[1] += 1
        # E --> I
        for exp in e_list:
            if exp[2] <= exp[1]:  # abbiamo superato la durata dell'esposizione generata
                duration_gamma = random.expovariate(gamma1)
                i_list.append([exp[0], 0, duration_gamma])
                e_list.remove(exp)
            else:
                exp[1] += 1

        for elem in i_list:
            ngbs = graph.neighbors(elem[0])
            for ngb in ngbs:
                if ngb in s_list:
                    r = random.uniform(0.0, 1.0)
                    if r < beta1:  # l'infetto contatta il vicino
                        duration_sigma = random.expovariate(sigma)
                        e_list.append([ngb, 0, duration_sigma])
                        s_list.remove(ngb)

    return [s_t, i_t, r_t]


def sim_SEIR_v2(graph, interval_tot):
    global s_list
    global i_list
    global r_list
    global s_t
    global i_t
    global r_t
    gamma1 = gamma * (1 / step_p_day)
    sigma1 = sigma * (1 / step_p_day)
    beta1 = beta * (1 / step_p_day)
    for elem in i_list:
        elem[2] = random.expovariate(gamma)  # 1 / gamma
        # print(elem[2])
    print("\nGRAFO IN ESAME")
    print("Nodi del grafo: " + str(list(graph)))
    # print("archi del grafo: " + str(list(graph.edges)))
    print("\nbefor SEIR")
    print(" S totali: " + str(sorted(s_list)))
    print(" E totali: " + str(e_list))
    print(" I totali: " + str(i_list))
    print(" R totali: " + str(r_list) + "")
    # simulazione
    for step in range(0, interval_tot):
        if (step % step_p_day) == 0:
            print(i_list)
            s_t.append(len(s_list))
            e_t.append(len(e_list))
            i_t.append(len(i_list))
            r_t.append(len(r_list))

        # I --> R
        for infect in i_list:
            if infect[2] <= infect[1]:  # abbiamo superato la durata dell'infezione generata
                r_list.append(infect[0])
                i_list.remove(infect)
            else:
                infect[1] += 1
        # E --> I
        for exp in e_list:
            if exp[2] <= exp[1]:  # abbiamo superato la durata dell'esposizione generata
                duration_gamma = random.expovariate(gamma1)
                i_list.append([exp[0], 0, duration_gamma])
                e_list.remove(exp)
            else:
                exp[1] += 1

        for elem in list(graph):
            if is_infected(elem) != -1:  # se il nodo è infetto
                ngbs = graph.neighbors(elem)
                for ngb in ngbs:
                    r = random.uniform(0.0, 1.0)
                    if r < beta1 and ngb in s_list:
                        duration_sigma = random.expovariate(sigma)
                        e_list.append([ngb, 0, duration_sigma])
                        s_list.remove(ngb)

    print("SEIR result after " + str(interval_tot) + " step:")
    print(" S totali: " + str(sorted(s_list)))
    print(" E totali: " + str(e_list))
    print(" I totali: " + str(i_list))
    print(" R totali: " + str(r_list))
    print("---" * 20)
    return [s_t, i_t, r_t]


# def simulation_SIR(graph, n_inf):
#     # inizializzazione
#     global i_list
#     global s_list
#     global r_list
#     s_t = []
#     i_t = []
#     r_t = []
#     mu = 0.2
#     beta = 0.1
#     list_nodes = list(graph.nodes())
#     random.shuffle(list_nodes)
#     i_list = [[elem, 1] for elem in list_nodes[0:n_inf]]
#     s_list = list_nodes[n_inf:]
#     r_list = []
#     s_t.append(len(s_list))
#     i_t.append(len(i_list))
#     r_t.append(len(r_list))
#
#     # simulazione
#     for time in range(0, 30):
#         # I -> R
#         temp = []
#         for elem in i_list:
#             rv = random.normalvariate(elem[1], 1.0)
#             if rv >= (1 / mu):
#                 r_list.append(elem[0])
#             else:
#                 elem[1] += 1
#                 temp.append(elem)
#         i_list = temp
#
#         for elem in i_list:
#             ngbs = graph.neighbors(elem[0])
#             for ngb in ngbs:
#                 r = random.uniform(0.0, 1.0)
#                 if r < beta and ngb in s_list:
#                     breakpoint()
#                     i_list.append([ngb, 1])
#                     s_list.remove(ngb)
#         s_t.append(len(s_list))
#         i_t.append(len(i_list))
#         r_t.append(len(r_list))
#
#     print("s" + str(s_t))
#     print("i" + str(i_t))
#     print("r" + str(r_t))
#     return [s_t, i_t, r_t]


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
    print(time)
    print(s_t)
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
    plt.savefig("result_SEIR.png")
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

        print("Input Parameters: \n")
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
    # input();
    parse_input_file()
    input()
    initialize()

    # g7 = gm.create_school_graph([4, 6, 8, 9, 13, 23, 45, 46, 47], 0.3)
    # print(gm.nx.edges(g7))
    # input()
    # simulate()
    # simulate()
    validation()
    plot_SEIR_result()
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

# def sim_SIR(graph, interval_tot):
#     global s_list
#     global i_list
#     global r_list
#     global s_t
#     global i_t
#     global r_t
#     mu = 1 / 15
#     sigma = 1 / 2
#     for elem in i_list:
#         elem[2] = random.expovariate(mu)  # 1 / mu
#         # print(elem[2])
#     print("\nGRAFO IN ESAME")
#     print("Nodi del grafo: " + str(list(graph)))
#     print("archi del grafo: " + str(list(graph.edges)))
#     print("\nbefor SIR")
#     print(" S totali: " + str(sorted(s_list)))
#     print(" I totali: " + str(i_list))
#     print(" R totali: " + str(r_list) + "")
#     # simulazione
#     for step in range(0, time + interval_tot):
#
#         s_t.append(len(s_list))
#         i_t.append(len(i_list))
#         r_t.append(len(r_list))
#
#         for infect in i_list:
#             if infect[2] <= infect[1]:  # abbiamo superato la durata dell'infezione generata
#                 r_list.append(infect[0])
#                 i_list.remove(infect)
#             else:
#                 infect[1] += 1
#
#         for elem in list(graph):
#             if is_infected(elem) != -1:  # se il nodo è infetto
#                 ngbs = graph.neighbors(elem)
#                 for ngb in ngbs:
#                     r = random.uniform(0.0, 1.0)
#                     if r < beta and ngb in s_list:
#                         duration = random.expovariate(mu)  # 1 / mu  # random.normalvariate(1 / mu, 1.0)
#                         # print(duration)
#                         # print("\n\n"+str(duration))
#                         i_list.append([ngb, 0, duration])
#                         s_list.remove(ngb)
#
#     print("SIR result after " + str(interval_tot) + " step:")
#     print(" S totali: " + str(sorted(s_list)))
#     print(" I totali: " + str(i_list))
#     print(" R totali: " + str(r_list))
#     print("---" * 20)
#     return [s_t, i_t, r_t]
