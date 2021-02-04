import csv


def write_csv(s_t, e_t, i_t, r_t, col):
    st_file = open('output/s_t.csv', 'a')
    et_file = open('output/e_t.csv', 'a')
    it_file = open('output/i_t.csv', 'a')
    rt_file = open('output/r_t.csv', 'a')
    with st_file:
        writer = csv.writer(st_file)
        writer.writerow(s_t)
    with et_file:
        writer = csv.writer(et_file)
        writer.writerow(e_t)
    with it_file:
        writer = csv.writer(it_file)
        writer.writerow(i_t)
    with rt_file:
        writer = csv.writer(rt_file)
        writer.writerow(r_t)
    print("Files have been written")


def calculate_average_from_csv():
    st_file = open('output/s_t.csv', 'r')
    et_file = open('output/e_t.csv', 'r')
    it_file = open('output/i_t.csv', 'r')
    rt_file = open('output/r_t.csv', 'r')
    st_avg = []
    et_avg = []
    it_avg = []
    rt_avg = []
    with st_file:
        # reader_1 = csv.reader(st_file, delimiter='\t')
        reader = csv.reader(st_file)
        size = len(next(reader))  # Read first line and count columns
        st_file.seek(0)  # go back to beginning of file
        st_avg = [0 for elem in range(0, size)]
        for row in reader:
            # print(row)
            st_avg = [st_avg[i] + int(row[i]) for i in range(0, size)]
        for index in range(0, size):
            st_avg[index] = st_avg[index] / size
        # print("somme")
        # print(st_avg)
    with et_file:
        et_avg = [0 for elem in range(0, size)]
        reader = csv.reader(et_file)
        for row in reader:
            # print(row)
            et_avg = [et_avg[i] + int(row[i]) for i in range(0, size)]
        for index in range(0, size):
            et_avg[index] = et_avg[index] / size
        # print("somme")
        # print(et_avg)

    with it_file:
        it_avg = [0 for elem in range(0, size)]
        reader = csv.reader(it_file)
        for row in reader:
            # print(row)
            it_avg = [it_avg[i] + int(row[i]) for i in range(0, size)]
        for index in range(0, size):
            it_avg[index] = it_avg[index] / size
        # print("somme")
        # print(it_avg)
    with rt_file:
        rt_avg = [0 for elem in range(0, size)]
        reader = csv.reader(rt_file)
        for row in reader:
            # print(row)
            rt_avg = [rt_avg[i] + int(row[i]) for i in range(0, size)]
        for index in range(0, size):
            rt_avg[index] = rt_avg[index] / size
        # print("somme")
    return [st_avg, et_avg, it_avg, rt_avg]


def clear_csv():
    st_file = open('output/s_t.csv', 'w+')
    et_file = open('output/e_t.csv', 'w+')
    it_file = open('output/i_t.csv', 'w+')
    rt_file = open('output/r_t.csv', 'w+')

    print("Files have been cleared")
