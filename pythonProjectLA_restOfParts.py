import copy
import math
import json
import random

import numpy as np
import networkx as nx
import scipy as sp
import self as self
from numpy import sort
from scipy.integrate import RK45
from matplotlib import pyplot as plt
from math import e
from scipy import integrate as inte
import math
from math import e


class Graph:
    # list of frequencies
    nodes_frequency = []
    # list of edges (as list like [1,2])
    adjacency_matrix = []
    naturalFreq = []
    vertices = []

    def __init__(self, nodes_frequency, adjacency_matrix, natural_frequency):
        self.nodes_frequency = nodes_frequency
        self.adjacency_matrix = adjacency_matrix
        self.naturalFreq = natural_frequency
        self.vertices = list(range(len(adjacency_matrix)))

    def number_of_nodes(self):
        return len(self.nodes_frequency)

    def get_mean_frequency(self):
        sum = 0.0
        for i in range(len(self.nodes_frequency)):
            sum += self.nodes_frequency[i]
        sum /= len(self.nodes_frequency)
        return sum
        pass

    def centroid_frequencies(self):
        average = self.get_mean_frequency()

        for i in range(len(self.nodes_frequency)):
            self.nodes_frequency[i] -= average

        pass

    def r(self):
        pass

    def is_edge(self, x, y):
        return self.adjacency_matrix[x][y] == 1

    def add_edge(self, x, y):
        if not self.is_edge(x, y):
            self.adjacency_matrix[x][y] = 1
            self.adjacency_matrix[y][x] = 1

    def remove_edge(self, x, y):
        if self.is_edge(x, y):
            self.adjacency_matrix[x][y] = 0
            self.adjacency_matrix[y][x] = 0

    def normalize_frequencies(self):
        for i in range(len(self.nodes_frequency)):
            while self.nodes_frequency[i] > math.pi:
                self.nodes_frequency[i] -= math.pi


graph_json = json.load(open('graphs.json', 'r'))
graph = Graph(graph_json[0], graph_json[1], graph_json[2])

print(graph.get_mean_frequency())
graph.centroid_frequencies()
print(graph.get_mean_frequency())


def r(graph):
    # Graph.r = 1
    sum = 0
    number = 0
    for i in range(len(graph.nodes_frequency)):
        complexNumber = complex(number, graph.nodes_frequency[i])
        sum += e ** complexNumber
    sum /= len(graph.nodes_frequency)
    return math.sqrt(sum.imag ** 2 + sum.real ** 2)


pass

Graph.r = r
print(graph.r())


def theta_calculator(graph, k):
    newArr = []
    for i in range(len(graph.nodes_frequency)):
        sum = 0
        for j in range(len(graph.nodes_frequency)):
            sum += (graph.adjacency_matrix[i][j] * np.sin(graph.nodes_frequency[j] - graph.nodes_frequency[i]))

        sum *= k
        newArr.append(sum + graph.naturalFreq[i])
    return newArr
    pass


print(sorted(theta_calculator(graph, 1)))


def next_step(graph, k, t0=0.1, t_bound=10, rtol=0.001, atol=1e-06):
    solution = RK45(lambda t, y: theta_calculator(graph, k), t0, graph.nodes_frequency, t_bound, 1e-06, 1e-06, rtol,
                    atol)
    solution.step()
    t_values = []
    y_values = []

    t_values.append(solution.t)
    y_values = solution.y

    return Graph(y_values, graph.adjacency_matrix, graph.naturalFreq)

    pass


print(graph.nodes_frequency)
print(next_step(graph, 1).nodes_frequency)


def plot_rn(graph_main, k, n):
    t = 0
    graph = copy.deepcopy(graph_main)
    t_values = []
    r_values = []
    t_values.append(0)
    r_values.append(graph.r())
    for i in range(n):
        graph = next_step(graph, k, t, 10, 0.001, 1e-06)
        r_values.append(graph.r())
        t += 1e-06
        t_values.append(t)

    plt.plot(t_values, r_values)
    plt.show()

    pass

plot_rn(graph , 1 , 10)


def checkMatrixCell(flag, randomI, randomJ, saveI, saveJ):
    hold = True
    for i in range(flag + 1):
        hold = hold and ~(randomI == saveI[i] and randomJ == saveJ[i])
    return hold


def rewire(graph_main):
    graph = copy.deepcopy(graph_main)
    saveI = [None] * 4
    saveJ = [None] * 4
    flag = 0
    while (True):
        randomI = random.randint(0, len(graph.adjacency_matrix)-1)
        randomJ = random.randint(0, len(graph.adjacency_matrix)-1)

        if (graph.adjacency_matrix[randomI][randomJ] == 0 and checkMatrixCell(flag , randomI , randomJ , saveI , saveJ)):
            graph.adjacency_matrix[randomI][randomJ] = 1
            graph.adjacency_matrix[randomJ][randomI] = 1
            saveI[flag] = randomI
            saveJ[flag] = randomJ
            flag += 1
        if (flag == 2):
            break

    while (True):
        randomI = random.randint(0, len(graph.adjacency_matrix)-1)
        randomJ = random.randint(0, len(graph.adjacency_matrix)-1)
        if (graph.adjacency_matrix[randomI][randomJ] == 1 and checkMatrixCell(flag , randomI , randomJ , saveI , saveJ)):
            graph.adjacency_matrix[randomI][randomJ] = 0
            graph.adjacency_matrix[randomJ][randomI] = 0
            saveI[flag] = randomI
            saveJ[flag] = randomJ
            flag += 1
        if (flag == 4):
            break
    return graph
    pass


def accept_reject(graph, n, m, k):
    newGraphAdded = copy.deepcopy(graph)
    for i in range(n):
        holdMainGraph = copy.deepcopy(newGraphAdded)
        newGraph = rewire(newGraphAdded)
        holdNewGraph = copy.deepcopy(newGraph)
        t = 0
        for i in range(m):
            holdMainGraph = next_step(holdMainGraph, k, t, 10, 0.001, 1e-06)
            t += 1e-06
        t = 0
        for i in range(m):
            holdNewGraph = next_step(holdNewGraph, k, t, 10, 0.001, 1e-06)
            t += 1e-06

        if (holdNewGraph.r() > holdMainGraph.r()):
            newGraphAdded = newGraph

    return newGraphAdded
    pass


graph_rewired = accept_reject(copy.deepcopy(graph), 100, 10, 0.5)
plot_rn(graph_rewired, 0.5, 20)
plot_rn(graph, 0.5, 20)
