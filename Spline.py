# La Grange
import pandas as pd
import numpy as np
from matplotlib import pyplot


def load_file(file_name, nth_element):

    read = pd.read_csv(file_name)
    data = np.array(read)
    array_length = len(data)
    element = np.array(data[array_length-1])
    data = data[0:array_length-2:nth_element] # ready data
    data = np.vstack((data, element))
    return data


def my_back_solve(matr, vec):

    var = np.zeros(len(vec))
    for i in range(len(vec) - 1, -1, -1):
        tmp = vec[i]
        for j in range(len(vec) - 1, i, -1):
            tmp -= var[j] * matr.item(i, j)
        var[i] = tmp / matr.item(i, i)

    return var


def my_solve(matr, vec):

    var = np.zeros(len(vec))
    var[0] = vec[0]/matr.item(0, 0)
    for i in range(1, len(vec)):
        wart = 0.0
        for j in range(0, i):
            wart += matr.item(i, j) * var[j]
        var[i] = (vec[i] - wart)/matr.item(i, i)

    return var


def solveLU(A, b):
    # using code designed for this task
    N = len(A)
    U = A.copy()
    L = np.eye(N)
    P = np.eye(N)

    for k in range(0, N - 1):

        # pivoting
        # find pivot
        ind = np.argmax(abs(U[k:N, k]))
        ind = ind + k

        # interchange rows
        U[[k, ind], k:N] = U[[ind, k], k:N]
        L[[k, ind], 0:k - 1] = L[[ind, k], 0:k - 1]
        P[[k, ind], :] = P[[ind, k], :]

        # using my last project
        for j in range(k + 1, N):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:N] = U[j, k:N] - L[j, k] * U[k, k:N]

    b = np.dot(P, b)
    y = my_solve(L, b)
    x = my_back_solve(U, y)

    return x


def spline(points):

    x = np.array(points[:, 0])
    y = np.array(points[:, 1])

    # ai = yi

    a = np.delete(y, (len(y)-1), axis=0)

    # as I've already computed 'a' factor, I have 3 more factors to be computed: b, c, d. I need: 3(b,c,d)
    # times n-1 (n points - n-1 functions) equations ----

    number_of_equations = (len(x) - 1)*3
    A = np.zeros((number_of_equations, number_of_equations))
    b_vec = np.zeros(number_of_equations)

    for i in range(0, number_of_equations, 3):

        # h = (xi+1 - xi)

        h = x[i//3+1] - x[i//3]

        # bi*h + ci*h^2 + di*h^3 = yi+1 - yi
        # this equation will be placed on every 3rd row of A matrix
        # my 'b' will be placed on every 3rd column of A matrix
        # my 'c' will be placed on every 3rd + 1 column of A matrix
        # my 'd' will be placed on every 3rd + 2 column of A matrix

        A[i, i] = h
        A[i, (i+1)] = pow(h, 2)
        A[i, (i+2)] = pow(h, 3)
        b_vec[i] = y[i//3+1]-y[i//3]

        if i < (number_of_equations-3):

            # bi + 2*h*ci + 3*h^2*di - bi+1 = 0
            # this equation will be placed on every 3rd + 1 row of A matrix

            A[i+1, i] = 1
            A[i+1, (i+1)] = 2*h
            A[i+1, (i+2)] = 3*pow(h, 2)
            A[i+1, (i+3)] = -1
            # b_vec[i+1] = 0, does not bring any change

            # 2*ci + 6*h*di - 2*ci+1 = 0
            # this equation will be placed on every 3rd + 2 row of A matrix

            A[i + 2, (i + 4)] = -2
            A[i+2, (i+1)] = 2
            A[i+2, (i+2)] = 6*h

            # b_vec[i+1] = 0, does not bring any change

        # marginal case

        else:

            # c0 = 0, 2*ci + 6*h*di = 0

            A[i+1, 1] = 1
            A[i+2, (i + 1)] = 2  # TODO nie wiem czy tak
            A[i+2, (i + 2)] = 6*h

    result = solveLU(A, b_vec)

    def equation(_x):
        if x[0] <= _x <= x[-1]:
            for k in range(0, len(points)):
                xi, yi = points[k]
                xj, yj = points[k + 1]
                if float(xi) <= _x <= float(xj):
                    b = result[k*3]
                    c = result[k*3 + 1]
                    d = result[k*3 + 2]
                    h = _x - float(xi)
                    return a[k] + b * h + c * pow(h, 2) + d * pow(h, 3)
    return equation


# ---------- MOUNT EVEREST, EVERY 5TH POINT --------------


ddd = np.array([[1,6],[3,-2],[5,4]])
readySpline = spline(ddd)

# data preparation
# raw
raw_data = pd.read_csv("2018_paths/MountEverest.csv")
raw_data = np.array(raw_data)
# every 5th
data_m_5 = load_file("2018_paths/MountEverest.csv", 5)
readySpline = spline(data_m_5)


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
    height_m_5_after_interpolation[i] = (readySpline(x))

pyplot.figure()
pyplot.plot(distance, height, 'y.', label='wszystkie dane')
pyplot.plot(distance_m_5, height_m_5, 'r.', label='węzły do interpolacji: co 5. punkt')
pyplot.plot(distance, height_m_5_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Mount Everest')
pyplot.title('wykorzystująca funkcje sklejane trzeciego stopnia (co 5. punkt)', fontsize=9)
pyplot.ylabel('Wysokość [m]')
pyplot.xlabel('Odległość od początku trasy [m]')
pyplot.legend()
pyplot.grid()
pyplot.show()


# ---------- MOUNT EVEREST, EVERY 20TH POINT --------------

# data preparation
# every 20th
data_m_20 = load_file("2018_paths/MountEverest.csv", 20)
readySpline = spline(data_m_20)


# calculating on raw data
height = np.empty(len(raw_data))
distance = np.empty(len(raw_data))

for i in range(len(raw_data)):
    x, y = raw_data[i]
    height[i] = y
    distance[i] = x


# calculating on every 20th element of raw data
distance_m_20 = np.empty(len(data_m_20))
height_m_20 = np.empty(len(data_m_20))
height_m_20_after_interpolation = np.empty(len(raw_data))

for i in range(len(data_m_20)):
    x, y = data_m_20[i]
    height_m_20[i] = y
    distance_m_20[i] = x

for i in range(len(raw_data)):
    x = distance[i]
    height_m_20_after_interpolation[i] = (readySpline(x))

pyplot.figure()
pyplot.plot(distance, height, 'y.', label='wszystkie dane')
pyplot.plot(distance_m_20, height_m_20, 'r.', label='węzły do interpolacji: co 20. punkt')
pyplot.plot(distance, height_m_20_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Mount Everest')
pyplot.title('wykorzystująca funkcje sklejane trzeciego stopnia (co 20. punkt)', fontsize=9)
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
readySpline = spline(data_sg_5)


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
    height_sg_5_after_interpolation[i] = (readySpline(x))

pyplot.figure()
pyplot.plot(distance, height, 'y.', label='wszystkie dane')
pyplot.plot(distance_sg_5, height_sg_5, 'r.', label='węzły do interpolacji: co 5. punkt')
pyplot.plot(distance, height_sg_5_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Spacerniak Gdański')
pyplot.title('wykorzystująca funkcje sklejane trzeciego stopnia (co 5. punkt)', fontsize=9)
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
readySpline = spline(data_gc_5)


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
    height_gc_5_after_interpolation[i] = (readySpline(x))

pyplot.figure()
pyplot.plot(distance, height, 'y.', label='wszystkie dane')
pyplot.plot(distance_gc_5, height_gc_5, 'r.', label='węzły do interpolacji: co 5. punkt')
pyplot.plot(distance, height_gc_5_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Głębia Challengera')
pyplot.title('wykorzystująca funkcje sklejane trzeciego stopnia (co 5. punkt)', fontsize=9)
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
readySpline = spline(data_kk_5)


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
    height_kk_5_after_interpolation[i] = (readySpline(x))

pyplot.figure()
pyplot.plot(distance, height, 'y.', label='wszystkie dane')
pyplot.plot(distance_kk_5, height_kk_5, 'r.', label='węzły do interpolacji: co 5. punkt')
pyplot.plot(distance, height_kk_5_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Wielki Kanion Kolorado')
pyplot.title('wykorzystująca funkcje sklejane trzeciego stopnia (co 5. punkt)', fontsize=9)
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
data_sg_n_3 = np.vstack((data_1, data_2))
readySpline = spline(data_sg_n_3)


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
    height_sg_n_3_after_interpolation[i] = (readySpline(x))

pyplot.figure()
pyplot.plot(distance, height, 'y.', label='wszystkie dane')
pyplot.plot(distance_sg_n_3, height_sg_n_3, 'r.', label='węzły do interpolacji: co 15. punkt, potem co 3. punkt')
pyplot.plot(distance, height_sg_n_3_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Spacerniak Gdański')
pyplot.title('wykorzystująca funkcje sklejane trzeciego stopnia (co 15. punkt, potem co 3. punkt)', fontsize=9)
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

readySpline = spline(data_sg_n_3_2)


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
    height_sg_n_3_2_after_interpolation[i] = (readySpline(x))

pyplot.figure()
pyplot.plot(distance, height, 'y.', label='wszystkie dane')
pyplot.plot(distance_sg_n_3_2, height_sg_n_3_2, 'r.', label='węzły do interpolacji: co 3. punkt, potem co 10. punkt, potem co 3. punkt')
pyplot.plot(distance, height_sg_n_3_2_after_interpolation, color='black', label='funkcja interpolująca')
pyplot.suptitle('Aproksymacja profilu wysokościowego Spacerniak Gdański')
pyplot.title('wykorzystująca funkcje sklejane trzeciego stopnia (co 3. punkt, potem co 10. punkt, potem co 3. punkt)', fontsize=9)
pyplot.ylabel('Wysokość [m]')
pyplot.xlabel('Odległość od początku trasy [m]')
pyplot.legend()
pyplot.grid()
pyplot.show()

