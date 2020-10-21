# La Grange
import pandas as pd
import numpy as np
from matplotlib import pyplot


def load_file(file_name, nth_element):
    read = pd.read_csv(file_name)
    data = np.array(read)
    array_length = len(data)
    element = np.array(data[array_length - 1])
    data = data[0:array_length - 2:nth_element]  # ready data
    data = np.vstack((data, element))
    return data



def laGrange(points):
    def equation(x):
        sum = 0
        length = len(points)

        for i in range(length):
            first = 1.0
            for j in range(length):
                if (i != j):
                    x_i, y_i = points[i]
                    x_j, y_j = points[j]
                    first = first * (x - x_j) / (x_i - x_j)
            sum = sum + first * y_i
        return sum
    return equation


# ---------- MOUNT EVEREST, EVERY 5TH POINT --------------


# data preparation
# raw
raw_data = pd.read_csv("2018_paths/MountEverest.csv")
raw_data = np.array(raw_data)
# every 5th
data_m_5 = load_file("2018_paths/MountEverest.csv", 5)
readyLaGrange = laGrange(data_m_5)


# calculating on raw data
height = np.empty(len(raw_data))
distance = np.empty(len(raw_data))

for i in range(len(raw_data)):
    x, y = raw_data[i]
    height[i] = y
    distance[i] = x


# calculating on every 5th element of raw data
distance_m_5 = np.empty(len(data_m_5))
height_m_5 = np.empty(len(data_m_5))
height_m_5_after_interpolation = np.empty(len(raw_data))

for i in range(len(data_m_5)):
    x, y = data_m_5[i]
    height_m_5[i] = y
    distance_m_5[i] = x

for i in range(len(raw_data)):
    x = distance[i]
    height_m_5_after_interpolation[i] = (readyLaGrange(x))

pyplot.figure()
pyplot.semilogy(distance, height, 'y.', label='wszystkie dane')
pyplot.semilogy(distance_m_5, height_m_5, 'r.', label='węzły do interpolacji: co 5. punkt')
pyplot.semilogy(distance, height_m_5_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Mount Everest')
pyplot.title('wykorzystująca wielomian interpolacyjny Lagrange’a (co 5. punkt)', fontsize=9)
pyplot.ylabel('Wysokość [m]')
pyplot.xlabel('Odległość od początku trasy [m]')
pyplot.legend()
pyplot.grid()
pyplot.show()


# ---------- MOUNT EVEREST, EVERY 20TH POINT --------------

# data preparation
# every 20th
data_m_20 = load_file("2018_paths/MountEverest.csv", 20)
readyLaGrange = laGrange(data_m_20)


# calculating on raw data
height = np.empty(len(raw_data))
distance = np.empty(len(raw_data))

for i in range(len(raw_data)):
    x, y = raw_data[i]
    height[i] = y
    distance[i] = x


# calculating on every 5th element of raw data
distance_m_20 = np.empty(len(data_m_20))
height_m_20 = np.empty(len(data_m_20))
height_m_20_after_interpolation = np.empty(len(raw_data))

for i in range(len(data_m_20)):
    x, y = data_m_20[i]
    height_m_20[i] = y
    distance_m_20[i] = x

for i in range(len(raw_data)):
    x = distance[i]
    height_m_20_after_interpolation[i] = (readyLaGrange(x))

pyplot.figure()
pyplot.semilogy(distance, height, 'y.', label='wszystkie dane')
pyplot.semilogy(distance_m_20, height_m_20, 'r.', label='węzły do interpolacji: co 20. punkt')
pyplot.semilogy(distance, height_m_20_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Mount Everest')
pyplot.title('wykorzystująca wielomian interpolacyjny Lagrange’a (co 20. punkt)', fontsize=9)
pyplot.ylabel('Wysokość [m]')
pyplot.xlabel('Odległość od początku trasy [m]')
pyplot.legend()
pyplot.grid()
pyplot.show()





# ---------- SPACERNIAK GDAŃSK, EVERY 5TH POINT --------------

# data preparation
raw_data = pd.read_csv("2018_paths/SpacerniakGdansk.csv")
raw_data = np.array(raw_data)
# every 5th
data_sg_5 = load_file("2018_paths/SpacerniakGdansk.csv", 5)
readyLaGrange = laGrange(data_sg_5)


# calculating on raw data
height = np.empty(len(raw_data))
distance = np.empty(len(raw_data))

for i in range(len(raw_data)):
    x, y = raw_data[i]
    height[i] = y
    distance[i] = x


# calculating on every 5th element of raw data
distance_sg_5 = np.empty(len(data_sg_5))
height_sg_5 = np.empty(len(data_sg_5))
height_sg_5_after_interpolation = np.empty(len(raw_data))

for i in range(len(data_sg_5)):
    x, y = data_sg_5[i]
    height_sg_5[i] = y
    distance_sg_5[i] = x

for i in range(len(raw_data)):
    x = distance[i]
    height_sg_5_after_interpolation[i] = (readyLaGrange(x))

pyplot.figure()
pyplot.semilogy(distance, height, 'y.', label='wszystkie dane')
pyplot.semilogy(distance_sg_5, height_sg_5, 'r.', label='węzły do interpolacji: co 5. punkt')
pyplot.semilogy(distance, height_sg_5_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Spacerniak Gdański')
pyplot.title('wykorzystująca wielomian interpolacyjny Lagrange’a (co 5. punkt)', fontsize=9)
pyplot.ylabel('Wysokość [m]')
pyplot.xlabel('Odległość od początku trasy [m]')
pyplot.legend()
pyplot.grid()
pyplot.show()



# ---------- GŁĘBIA CHALLENGERA, EVERY 5TH POINT --------------

# data preparation
raw_data = pd.read_csv("2018_paths/GlebiaChallengera.csv")
raw_data = np.array(raw_data)
# every 5th
data_gc_5 = load_file("2018_paths/GlebiaChallengera.csv", 5)
readyLaGrange = laGrange(data_gc_5)


# calculating on raw data
height = np.empty(len(raw_data))
distance = np.empty(len(raw_data))

for i in range(len(raw_data)):
    x, y = raw_data[i]
    height[i] = y
    distance[i] = x


# calculating on every 5th element of raw data
distance_gc_5 = np.empty(len(data_gc_5))
height_gc_5 = np.empty(len(data_gc_5))
height_gc_5_after_interpolation = np.empty(len(raw_data))

for i in range(len(data_gc_5)):
    x, y = data_gc_5[i]
    height_gc_5[i] = y
    distance_gc_5[i] = x

for i in range(len(raw_data)):
    x = distance[i]
    height_gc_5_after_interpolation[i] = (readyLaGrange(x))

pyplot.figure()
pyplot.semilogy(distance, height, 'y.', label='wszystkie dane')
pyplot.semilogy(distance_gc_5, height_gc_5, 'r.', label='węzły do interpolacji: co 5. punkt')
pyplot.semilogy(distance, height_gc_5_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Głębia Challengera')
pyplot.title('wykorzystująca wielomian interpolacyjny Lagrange’a (co 5. punkt)', fontsize=9)
pyplot.ylabel('Wysokość [m]')
pyplot.xlabel('Odległość od początku trasy [m]')
pyplot.legend()
pyplot.grid()
pyplot.show()



# ---------- WIELKI KANION KOLORADO, EVERY 5TH POINT --------------

# data preparation
raw_data = pd.read_csv("2018_paths/WielkiKanionKolorado.csv")
raw_data = np.array(raw_data)
# every 5th
data_kk_5 = load_file("2018_paths/WielkiKanionKolorado.csv", 5)
readyLaGrange = laGrange(data_kk_5)


# calculating on raw data
height = np.empty(len(raw_data))
distance = np.empty(len(raw_data))

for i in range(len(raw_data)):
    x, y = raw_data[i]
    height[i] = y
    distance[i] = x


# calculating on every 5th element of raw data
distance_kk_5 = np.empty(len(data_kk_5))
height_kk_5 = np.empty(len(data_kk_5))
height_kk_5_after_interpolation = np.empty(len(raw_data))

for i in range(len(data_kk_5)):
    x, y = data_kk_5[i]
    height_kk_5[i] = y
    distance_kk_5[i] = x

for i in range(len(raw_data)):
    x = distance[i]
    height_kk_5_after_interpolation[i] = (readyLaGrange(x))

pyplot.figure()
pyplot.semilogy(distance, height, 'y.', label='wszystkie dane')
pyplot.semilogy(distance_kk_5, height_kk_5, 'r.', label='węzły do interpolacji: co 5. punkt')
pyplot.semilogy(distance, height_kk_5_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Wielki Kanion Kolorado')
pyplot.title('wykorzystująca wielomian interpolacyjny Lagrange’a (co 5. punkt)', fontsize=9)
pyplot.ylabel('Wysokość [m]')
pyplot.xlabel('Odległość od początku trasy [m]')
pyplot.legend()
pyplot.grid()
pyplot.show()






# ---------- SPACERNIAK GDAŃSK, NOT EVERY 3TH POINT --------------

# data preparation
raw_data = pd.read_csv("2018_paths/SpacerniakGdansk.csv")
raw_data = np.array(raw_data)

read = pd.read_csv("2018_paths/SpacerniakGdansk.csv")
data = np.array(read)
array_length = len(data)
element_2 = np.array(data[array_length - 1])
element_1 = np.array(data[359])
data_1 = data[0:358:15]
data_1 = np.vstack((data_1, element_1))
data_2 = data[360:array_length-2:3]
data_2 = np.vstack((data_2, element_2))
data_sg_n_3 = np.vstack((data_1,data_2))
readyLaGrange = laGrange(data_sg_n_3)


# calculating on raw data
height = np.empty(len(raw_data))
distance = np.empty(len(raw_data))

for i in range(len(raw_data)):
    x, y = raw_data[i]
    height[i] = y
    distance[i] = x


# calculating on every 5th element of raw data
distance_sg_n_3 = np.empty(len(data_sg_n_3))
height_sg_n_3 = np.empty(len(data_sg_n_3))
height_sg_n_3_after_interpolation = np.empty(len(raw_data))

for i in range(len(data_sg_n_3)):
    x, y = data_sg_n_3[i]
    height_sg_n_3[i] = y
    distance_sg_n_3[i] = x

for i in range(len(raw_data)):
    x = distance[i]
    height_sg_n_3_after_interpolation[i] = (readyLaGrange(x))

pyplot.figure()
pyplot.semilogy(distance, height, 'y.', label='wszystkie dane')
pyplot.semilogy(distance_sg_n_3, height_sg_n_3, 'r.', label='węzły do interpolacji: co 15. punkt, potem co 3. punkt')
pyplot.semilogy(distance, height_sg_n_3_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Spacerniak Gdański')
pyplot.title('wykorzystująca wielomian interpolacyjny Lagrange’a (co 15. punkt, potem co 3. punkt)', fontsize=9)
pyplot.ylabel('Wysokość [m]')
pyplot.xlabel('Odległość od początku trasy [m]')
pyplot.legend()
pyplot.grid()
pyplot.show()




# ---------- SPACERNIAK GDAŃSK, NOT EVERY 3TH POINT --------------

# data preparation
raw_data = pd.read_csv("2018_paths/SpacerniakGdansk.csv")
raw_data = np.array(raw_data)

read = pd.read_csv("2018_paths/SpacerniakGdansk.csv")
data = np.array(read)
array_length = len(data)
element_3 = np.array(data[array_length - 1])
element_2 = np.array(data[264])
element_1 = np.array(data[209])
data_1 = data[0:208:3]
data_1 = np.vstack((data_1, element_1))
data_2 = data[210:263:10]
data_2 = np.vstack((data_2, element_2))
data_3 = data[265:array_length-2:3]
data_3 = np.vstack((data_3, element_3))
data_sg_n_3 = np.vstack((data_1,data_2))
data_sg_n_3_2 = np.vstack((data_sg_n_3,data_3))

readyLaGrange = laGrange(data_sg_n_3_2)


# calculating on raw data
height = np.empty(len(raw_data))
distance = np.empty(len(raw_data))

for i in range(len(raw_data)):
    x, y = raw_data[i]
    height[i] = y
    distance[i] = x


# calculating on every 5th element of raw data
distance_sg_n_3_2 = np.empty(len(data_sg_n_3_2))
height_sg_n_3_2 = np.empty(len(data_sg_n_3_2))
height_sg_n_3_2_after_interpolation = np.empty(len(raw_data))

for i in range(len(data_sg_n_3_2)):
    x, y = data_sg_n_3_2[i]
    height_sg_n_3_2[i] = y
    distance_sg_n_3_2[i] = x

for i in range(len(raw_data)):
    x = distance[i]
    height_sg_n_3_2_after_interpolation[i] = (readyLaGrange(x))

pyplot.figure()
pyplot.semilogy(distance, height, 'y.', label='wszystkie dane')
pyplot.semilogy(distance_sg_n_3_2, height_sg_n_3_2, 'r.', label='węzły do interpolacji: co 3. punkt, potem co 10. punkt, potem co 3. punkt')
pyplot.semilogy(distance, height_sg_n_3_2_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Spacerniak Gdański')
pyplot.title('wykorzystująca wielomian interpolacyjny Lagrange’a (co 3. punkt, potem co 10. punkt, potem co 3. punkt)', fontsize=9)
pyplot.ylabel('Wysokość [m]')
pyplot.xlabel('Odległość od początku trasy [m]')
pyplot.legend()
pyplot.grid()
pyplot.show()

