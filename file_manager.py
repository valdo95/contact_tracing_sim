import csv
import yaml
import math


def get_r0_path_file():
    with open("config_files/path_files.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        return data["r_0_path"]


def get_r0_stat_path_file():
    with open("config_files/path_files.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        return data["r_0_stat_path"]


def get_path_files_seir():
    with open("config_files/path_files.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        return data["st_path"], data["et_path"], data["it_path"], data["rt_path"]


def get_path_avg_files_seir():
    with open("config_files/path_files.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        return data["avg_st_path"], data["avg_et_path"], data["avg_it_path"], data["avg_rt_path"]


def get_path_files_tracing():
    with open("config_files/path_files.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        return data["st_path"], data["et_path"], data["it_path"], data["rt_path"], data["ist_path"], data["qst_path"], \
               data["qeit_path"]


def get_path_files_tracing_queue():
    with open("config_files/path_files.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        return data["st_path"], data["et_path"], data["it_path"], data["rt_path"], data["ist_path"], data["qst_path"], \
               data["qeit_path"], data["wist_path"], data["wst_path"]


def get_path_avg_files_tracing():
    with open("config_files/path_files.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        return data["avg_st_path"], data["avg_et_path"], data["avg_it_path"], data["avg_rt_path"], data["avg_ist_path"], \
               data["avg_qst_path"], data["avg_qeit_path"]


def get_path_avg_files_tracing_queue():
    with open("config_files/path_files.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        return data["avg_st_path"], data["avg_et_path"], data["avg_it_path"], data["avg_rt_path"], data["avg_ist_path"], \
               data["avg_qst_path"], data["avg_qeit_path"], data["avg_wist_path"], data["avg_wst_path"], data[
                   "r_0_path"], data["r_0_stat_path"]


def calculate_average(file, size=-1, n_node=1):
    reader = csv.reader(file)
    if size == -1:
        size = len(next(reader))  # Read first line and count columns
        file.seek(0)  # go back to beginning of file
    st_avg = [0 for elem in range(0, size)]
    n_row = 0
    for row in reader:
        st_avg = [st_avg[i] + int(row[i]) for i in range(0, size)]
        n_row += 1
    for index in range(0, size):
        st_avg[index] = st_avg[index] / (n_row * n_node)
    return st_avg, size


def calculate_average_from_csv_seir():
    size = -1
    res = []
    for file_path in get_path_files_seir():
        # print(get_path_files_seir())
        with open(file_path, 'r') as file:
            [avg, size] = calculate_average(file, size)
            res.append(avg)
    return res


def calculate_average_from_csv_tracing(n_node=1):
    size = -1
    res = []
    for file_path in get_path_files_tracing():
        with open(file_path, 'r') as file:
            [avg, size] = calculate_average(file, size, n_node=n_node)
            res.append(avg)
    return res


def variance(data, mean, ddof=0):
    n = len(data)
    return sum((x - mean) ** 2 for x in data) / (n - ddof)


def stdev(data, mean, ddof=0):
    return math.sqrt(variance(data, mean, ddof))


def calculate_r0_avarage():
    file_path = get_r0_path_file()
    list_r0 = []
    avg_r0 = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            for elem in row:
                avg_r0 += int(elem)
                list_r0.append(int(elem))

    mean = avg_r0 / len(list_r0)
    sd = stdev(list_r0, mean, ddof=1)
    return mean, sd


def calculate_average_from_csv_tracing_queue(n_node=1):
    size = -1
    res = []
    for file_path in get_path_files_tracing_queue():
        with open(file_path, 'r') as file:
            [avg, size] = calculate_average(file, size, n_node=n_node)
            res.append(avg)
    return res


def clear_csv():
    for file in get_path_files_tracing_queue():
        f = open(file, "w+")
        f.close()
    print(str(file) + " have been cleared")


def clear_avg_csv():
    for file in get_path_avg_files_tracing_queue():
        f = open(file, "w+")
        f.close()
    print(str(file) + " have been cleared")


def read_csv(tracing=False, avg=False):
    if avg:
        if tracing:
            files = get_path_avg_files_tracing()
        else:
            files = get_path_avg_files_seir()
    else:
        if tracing:
            files = get_path_files_tracing()
        else:
            files = get_path_files_seir()
    res = []
    for file_path in files:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                for index in range(0, len(row)):
                    row[index] = float(row[index])
                res.append(row)
    return res


def write_csv_seir(s_t, e_t, i_t, r_t, avg=False):
    if avg:
        [st_path, et_path, it_path, rt_path] = get_path_avg_files_seir()
    else:
        [st_path, et_path, it_path, rt_path] = get_path_files_seir()
    with open(st_path, 'a') as st_file:
        writer = csv.writer(st_file)
        writer.writerow(s_t)
    with open(et_path, 'a') as et_file:
        writer = csv.writer(et_file)
        writer.writerow(e_t)
    with open(it_path, 'a') as it_file:
        writer = csv.writer(it_file)
        writer.writerow(i_t)
    with open(rt_path, 'a') as rt_file:
        writer = csv.writer(rt_file)
        writer.writerow(r_t)
    print("Files have been written")


def write_csv_tracing(s_t, e_t, i_t, r_t, is_t, qs_t, qei_t, avg=False):
    if avg:
        [st_path, et_path, it_path, rt_path, ist_path, qst_path, qeit_path] = get_path_avg_files_tracing()
    else:
        [st_path, et_path, it_path, rt_path, ist_path, qst_path, qeit_path] = get_path_files_tracing()
    with open(st_path, 'a') as st_file:
        writer = csv.writer(st_file)
        writer.writerow(s_t)
    with open(et_path, 'a') as et_file:
        writer = csv.writer(et_file)
        writer.writerow(e_t)
    with open(it_path, 'a') as it_file:
        writer = csv.writer(it_file)
        writer.writerow(i_t)
    with open(rt_path, 'a') as rt_file:
        writer = csv.writer(rt_file)
        writer.writerow(r_t)
    with open(ist_path, 'a') as ist_file:
        writer = csv.writer(ist_file)
        writer.writerow(is_t)
    with open(qst_path, 'a') as qst_file:
        writer = csv.writer(qst_file)
        writer.writerow(qs_t)
    with open(qeit_path, 'a') as qeit_file:
        writer = csv.writer(qeit_file)
        writer.writerow(qei_t)
    print("Files have been written")


def write_csv_r0(tagged_i):
    r0_path = get_r0_path_file()
    with open(r0_path, 'a') as r0_file:
        row = []
        for elem in tagged_i:
            row.append(elem[1])
        writer = csv.writer(r0_file)
        writer.writerow(row)


def write_statistic_r0(mean, sd):
    file_path = get_r0_stat_path_file()
    with open(file_path, 'a') as stat_file:
        writer = csv.writer(stat_file)
        writer.writerow([mean, sd])


def write_csv_tracing_queue(s_t, e_t, i_t, r_t, is_t, qs_t, qei_t, wis_t, ws_t, avg=False):
    if avg:
        [st_path, et_path, it_path, rt_path, ist_path, qst_path, qeit_path,
         wist_path, wst_path] = get_path_avg_files_tracing_queue()
    else:
        [st_path, et_path, it_path, rt_path, ist_path, qst_path, qeit_path, wist_path,
         wst_path] = get_path_files_tracing_queue()
    with open(st_path, 'a') as st_file:
        writer = csv.writer(st_file)
        writer.writerow(s_t)
    with open(et_path, 'a') as et_file:
        writer = csv.writer(et_file)
        writer.writerow(e_t)
    with open(it_path, 'a') as it_file:
        writer = csv.writer(it_file)
        writer.writerow(i_t)
    with open(rt_path, 'a') as rt_file:
        writer = csv.writer(rt_file)
        writer.writerow(r_t)
    with open(ist_path, 'a') as ist_file:
        writer = csv.writer(ist_file)
        writer.writerow(is_t)
    with open(qst_path, 'a') as qst_file:
        writer = csv.writer(qst_file)
        writer.writerow(qs_t)
    with open(qeit_path, 'a') as qeit_file:
        writer = csv.writer(qeit_file)
        writer.writerow(qei_t)
    with open(wist_path, 'a') as wist_file:
        writer = csv.writer(wist_file)
        writer.writerow(wis_t)
    with open(wst_path, 'a') as wst_file:
        writer = csv.writer(wst_file)
        writer.writerow(ws_t)
    print("Files have been written")
