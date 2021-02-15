import csv
import yaml


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
        return data["st_path"], data["et_path"], data["it_path"], data["rt_path"], data["ist_path"], data["qt_path"]


def get_path_avg_files_tracing():
    with open("config_files/path_files.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
        return data["avg_st_path"], data["avg_et_path"], data["avg_it_path"], data["avg_rt_path"], data["avg_ist_path"], \
               data["avg_qt_path"]


def calculate_average(file, size=-1):
    reader = csv.reader(file)
    if size == -1:
        size = len(next(reader))  # Read first line and count columns
        file.seek(0)  # go back to beginning of file
    st_avg = [0 for elem in range(0, size)]
    for row in reader:
        st_avg = [st_avg[i] + int(row[i]) for i in range(0, size)]
    for index in range(0, size):
        st_avg[index] = st_avg[index] / size
    return st_avg, size


def calculate_average_from_csv_seir():
    size = -1
    res = []
    for file_path in get_path_files_seir():
        with open(file_path, 'r') as file:
            [avg, size] = calculate_average(file, size)
            res.append(avg)
    return res


def calculate_average_from_csv_tracing():
    size = -1
    res = []
    for file_path in get_path_files_tracing():
        with open(file_path, 'r') as file:
            [avg, size] = calculate_average(file, size)
            res.append(avg)
    return res


def clear_csv():
    [st_path, et_path, it_path, rt_path, ist_path, qt_path] = get_path_files_tracing()
    st_file = open(st_path, 'w+')
    et_file = open(et_path, 'w+')
    it_file = open(it_path, 'w+')
    rt_file = open(rt_path, 'w+')
    ist_file = open(ist_path, "w+")
    qt_file = open(qt_path, "w+")

    print("Files have been cleared")


def clear_avg_csv():
    [st_path, et_path, it_path, rt_path, ist_path, qt_path] = get_path_avg_files_tracing()
    st_file = open(st_path, 'w+')
    et_file = open(et_path, 'w+')
    it_file = open(it_path, 'w+')
    rt_file = open(rt_path, 'w+')

    print("Files have been cleared")


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


def write_csv_tracing(s_t, e_t, i_t, r_t, is_t, q_t, avg=False):
    if avg:
        [st_path, et_path, it_path, rt_path, ist_path, qt_path] = get_path_avg_files_tracing()
    else:
        [st_path, et_path, it_path, rt_path, ist_path, qt_path] = get_path_files_tracing()
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
    with open(qt_path, 'a') as qt_file:
        writer = csv.writer(qt_file)
        writer.writerow(q_t)
    print("Files have been written")
