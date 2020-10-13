import graph_manager as gm
import random
import EoN as eon
import matplotlib.pyplot as plt
from itertools import islice
from random import randint

n = 50
i_list = []
s_list = []
r_list = []
i_t = []
s_t = []
r_t = []
people_tot = []
families = []
time = 0


def generate_families(min_size=1, max_size=6):
    it = iter(people_tot)
    while True:
        nxt = list(islice(it, randint(min_size, max_size)))
        if nxt:
            yield nxt
        else:
            break


def initialize(initial_I):
    global time
    global people_tot
    global families
    global i_list
    global s_list
    global r_list

    time = 0
    people_tot = [elem for elem in range(0, n)]
    i_list = [[elem, time] for elem in initial_I]
    for elem in people_tot:
        if is_infected(elem) == -1:
            s_list.append(elem)
    print(i_list)
    print("S " + str(s_list))
    # breakpoint()
    random.shuffle(people_tot)
    families = list(generate_families())
    r_list = []


def is_infected(elem):
    ctrl = True
    i = 0
    while ctrl and i < len(i_list):
        ctrl = not (elem == i_list[i][0])
        i += 1
    if ctrl:
        return -1
    return i - 1


def simulate():
    global step
    i = 0
    step = 5
    for elem in families:
        graph = gm.create_home_graph(elem)
        # gm.print_graph(graph, str(i))
        sim_SIR(graph, step)
        # input()


def sim_SIR(graph, interval_tot):
    global s_list
    global i_list
    global r_list
    global time
    mu = 1 / 6
    beta = 0.3
    s_t = []
    i_t = []
    r_t = []
    print("\nGRAFO IN ESAME")
    print("Nodi del grafo: " + str(list(graph)))
    print("archi del grafo: " + str(list(graph.edges)))
    print("\nbefor SIR")
    print(" S totali: " + str(s_list))
    print(" I totali: " + str(i_list))
    print(" R totali: " + str(r_list) + "")
    # simulazione
    for time in range(time, time + interval_tot):
        # I -> R
        temp = []
        for elem in list(graph):
            index = is_infected(elem)
            if index != -1:
                rv = random.normalvariate(i_list[index][1], 1.0)
                if i_list[index][1] >= (1 / mu):
                    r_list.append(elem)
                else:
                    i_list[index][1] += 1
                    # temp.append(i_list[index][1])
                # i_list = temp

        for elem in list(graph):
            # print("elem " + str(elem))
            if is_infected(elem) != -1:
                ngbs = graph.neighbors(elem)
                # print("ngbs " + str(list(ngbs)))
                for ngb in ngbs:
                    r = random.uniform(0.0, 1.0)
                    if r < beta and ngb in s_list:
                        i_list.append([ngb, 0])
                        s_list.remove(ngb)
        s_t.append(len(s_list))
        i_t.append(len(i_list))
        r_t.append(len(r_list))

    # print("s" + str(s_t))
    # print("i" + str(i_t))
    # print("r" + str(r_t))
    print("SIR result after " + str(interval_tot) + " step:")
    print(" S totali: " + str(s_list))
    print(" I totali: " + str(i_list))
    print(" R totali: " + str(r_list))
    print("---" * 20)
    return [s_t, i_t, r_t]


def simulation_SIR(graph, n_inf):
    # inizializzazione
    global i_list
    global s_list
    global r_list
    s_t = []
    i_t = []
    r_t = []
    mu = 0.2
    beta = 0.3
    list_nodes = list(graph.nodes())
    random.shuffle(list_nodes)
    i_list = [[elem, 1] for elem in list_nodes[0:n_inf]]
    s_list = list_nodes[n_inf:]
    r_list = []
    s_t.append(len(s_list))
    i_t.append(len(i_list))
    r_t.append(len(r_list))

    # simulazione
    for time in range(0, 30):
        # I -> R
        temp = []
        for elem in i_list:
            rv = random.normalvariate(elem[1], 1.0)
            if rv >= (1 / mu):
                r_list.append(elem[0])
            else:
                elem[1] += 1
                temp.append(elem)
        i_list = temp

        for elem in i_list:
            ngbs = graph.neighbors(elem[0])
            for ngb in ngbs:
                r = random.uniform(0.0, 1.0)
                if r < beta and ngb in s_list:
                    breakpoint()
                    i_list.append([ngb, 1])
                    s_list.remove(ngb)
        s_t.append(len(s_list))
        i_t.append(len(i_list))
        r_t.append(len(r_list))

    print("s" + str(s_t))
    print("i" + str(i_t))
    print("r" + str(r_t))
    return [s_t, i_t, r_t]


def plot_SIR_result(s_t, i_t, r_t):
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


def epidemic(graph, n_inf):
    mu = 0.2
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


if __name__ == '__main__':
    initialize([4, 14, 2, 9])
    g7 = gm.create_school_graph([4, 6, 8, 9, 13, 23, 45, 46, 47], 0.3)
    print("fine funzione ")
    gm.write_graph(g7, "school")
    gm.print_graph(g7, "school")
    print(gm.nx.edges(g7))
    # input()
    # simulate()
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
