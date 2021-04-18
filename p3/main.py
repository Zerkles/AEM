import itertools

import tsplib95 as tsp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import random


class LocalSearchSolver:
    def __init__(self, g: nx.DiGraph, distance_matrix: np.matrix):
        self.g = g
        self.dm = distance_matrix
        self.g, self.start_node, self.unused_nodes = self.create_random_route(self.g)

    def __find_nn(self, node: int, nodes_list: list, dm: np.matrix):
        nn = nodes_list[0]
        min_distance = np.min(dm[nn - 1, node - 1])

        for n in nodes_list[1:]:
            distance = np.min(dm[n - 1, node - 1])

            if distance < min_distance:
                nn = n
                min_distance = distance

        return nn

    def __find_shortest_loop(self, node, edges, dm):
        n1_min, n2_min = edges[0]
        min_loop = dm[node - 1, n1_min - 1] + dm[node - 1, n2_min - 1] - dm[
            n1_min - 1, n2_min - 1]

        for e in edges[1:]:
            n1, n2 = e
            loop = dm[node - 1, n1 - 1] + dm[node - 1, n2 - 1] - dm[
                n1 - 1, n2 - 1]
            if loop < min_loop:
                min_loop = loop
                n1_min = n1
                n2_min = n2

        return n1_min, n2_min, min_loop

    def __replace_nodes(self, n1, n2):
        g = self.g
        n1_prev, n1_next = tuple(g.in_edges(n1))[0][0], tuple(g.out_edges(n1))[0][1]

        g.remove_edge(n1_prev, n1)
        g.remove_edge(n1, n1_next)

        g.add_edge(n1_prev, n2)
        g.add_edge(n2, n1_next)

    def __reorder_nodes(self, n1, n2):
        g = self.g
        n1_prev, n1_next = tuple(g.in_edges(n1))[0][0], tuple(g.out_edges(n1))[0][1]
        n2_prev, n2_next = tuple(g.in_edges(n2))[0][0], tuple(g.out_edges(n2))[0][1]

        if n1 == n2_prev:
            n1_next = n1
            n2_prev = n2

        g.remove_edge(n1_prev, n1)
        g.remove_edge(n1, n1_next)
        g.remove_edge(n2_prev, n2)
        g.remove_edge(n2, n2_next)

        g.add_edge(n1_prev, n2)
        g.add_edge(n2, n1_next)
        g.add_edge(n2_prev, n1)
        g.add_edge(n1, n2_next)

    def __reorder_edge(self, e1, e2):  # e1 początek krawędzi, e2 koniec
        g = self.g
        e1_prev = tuple(g.in_edges(e1))[0][0]
        e2_next = tuple(g.out_edges(e2))[0][1]

        g.remove_edge(e1_prev, e1)
        g.remove_edge(e2, e2_next)

        e_list = []
        e = e1
        while e != e2:
            e_list.append(list(g.out_edges(e))[0])
            e = tuple(g.out_edges(e))[0][1]

        for e, ee in e_list:
            g.remove_edge(e, ee)
            g.add_edge(ee, e)

        g.add_edge(e1_prev, e2)
        g.add_edge(e1, e2_next)

    def __delta_reorder_edge(self, e1, e2):
        g, dm = self.g, self.dm
        e1_prev = tuple(g.in_edges(e1))[0][0]
        e2_next = tuple(g.out_edges(e2))[0][1]

        old_distance = dm[e1_prev - 1, e1 - 1] + dm[e2 - 1, e2_next - 1]
        new_distance = dm[e1_prev - 1, e2 - 1] + dm[e1 - 1, e2_next - 1]

        return new_distance - old_distance

    def __delta_reorder_nodes(self, n1, n2):
        g, dm = self.g, self.dm

        n1_prev, n1_next = tuple(g.in_edges(n1))[0][0], tuple(g.out_edges(n1))[0][1]
        n2_prev, n2_next = tuple(g.in_edges(n2))[0][0], tuple(g.out_edges(n2))[0][1]

        if n1 == n2_prev:
            n1_next = n1
            n2_prev = n2

        old_distance = dm[n1_prev - 1, n1 - 1] + dm[n2_prev - 1, n2 - 1] + dm[n2 - 1, n2_next - 1] + dm[
            n1 - 1, n1_next - 1]
        new_distance = dm[n1_prev - 1, n2 - 1] + dm[n2_prev - 1, n1 - 1] + dm[n2 - 1, n1_next - 1] + dm[
            n1 - 1, n2_next - 1]

        return new_distance - old_distance

    def __delta_replace_nodes(self, n1, n2):  # n1 jest w ścieżce, n2 nie
        g, dm = self.g, self.dm

        n1_prev, n1_next = tuple(g.in_edges(n1))[0][0], tuple(g.out_edges(n1))[0][1]

        old_distance = dm[n1_prev - 1, n1 - 1] + dm[n1 - 1, n1_next - 1]
        new_distance = dm[n1_prev - 1, n2 - 1] + dm[n2 - 1, n1_next - 1]

        return new_distance - old_distance

    def _get_objective_function_value(self, g):
        dm = self.dm
        sum = 0
        for e in g.edges():
            u, v = e
            sum += dm[u - 1, v - 1]

        return sum

    def create_random_route(self, g):
        g.clear_edges()
        unused_nodes = list(g.nodes)
        start_node = node = random.choice(unused_nodes)
        unused_nodes.remove(start_node)

        while g.number_of_edges() != int(g.number_of_nodes() * 0.5) - 1:
            new_node = random.choice(unused_nodes)
            g.add_edge(node, new_node)
            unused_nodes.remove(new_node)
            node = new_node
        g.add_edge(node, start_node)

        self.show_graph(self.g, "random_route.png")
        return g, start_node, unused_nodes

    def random_alg(self):
        g = self.g
        g_copy = self.g.copy()
        minimum = self._get_objective_function_value(g)

        start_time = time.time()
        while time.time() < start_time + 5.39:
            g, _, _ = self.create_random_route(g)
            length = self._get_objective_function_value(g)
            if length < minimum:
                minimum = length

        print("Route length:", minimum)
        self.g = g_copy
        return g, minimum

    @staticmethod
    def update_moves_list(lst, val):
        for i in range(0, len(lst)):
            if val[2] < lst[i][2]:
                lst.insert(i, val)
                break
        return lst

    @staticmethod
    def remove_from_list(lst, val):
        removal_list = []
        for i in range(0, len(lst)):
            if lst[i][0] == val or lst[i][1] == val:
                removal_list.append(lst[i])
        return list(set(lst) - set(removal_list))

    def generate_moves(self, unused_nodes):
        g = self.g
        moves_edge, moves_node = [(0, 0, 0)], [(0, 0, 0)]
        # format: (wierzchołek1,wierzchołek2,wartość delty funkcji celu)

        used_nodes = list(set(g.nodes) - set(unused_nodes))
        # n1 = random.choice(used_nodes)
        for n1 in used_nodes:
            best_edge, best_node = (0, 0, 0), (0, 0, 0)
            n2 = tuple(g.out_edges(n1))[0][1]
            n1_prev = tuple(g.in_edges(n1))[0][0]
            while n2 != n1_prev:
                delta = self.__delta_reorder_edge(n1, n2)  # zamiana wierzchołków będących w grafie
                if best_edge[2] > delta and (n1, n2, delta) not in moves_edge:
                    best_edge = (n1, n2, delta)
                n2 = tuple(g.out_edges(n2))[0][1]
            moves_edge = self.update_moves_list(moves_edge, best_edge)
            # print(best_edge)

            for n2 in unused_nodes:
                if n1 == n2:
                    continue

                delta = self.__delta_replace_nodes(n1, n2)  # wyrzucenie wierzchołka i dodanie jednego spoza grafu
                if best_node[2] > delta and (n1, n2, delta) not in moves_node:
                    best_node = (n1, n2, delta)
            moves_node = self.update_moves_list(moves_node, best_node)

        # moves_node.remove((0, 0, 0))
        # moves_edge.remove((0, 0, 0))
        print("moves_node", moves_node)
        print("moves_edge", moves_edge)
        return moves_edge, moves_node

        # moves_edge, moves_node = [(0, 0, 0)] * 1000, [(0, 0, 0)] * 1000
        # g = self.g
        #
        # n1 = random.choice(list(set(g.nodes) - set(unused_nodes)))
        # n2 = tuple(g.out_edges(n1))[0][1]
        # n1_prev = tuple(g.in_edges(n1))[0][0]
        # while n2 != n1_prev:
        #     delta = self.__delta_reorder_edge(n1, n2)  # zamiana wierzchołków będących w grafie
        #     if delta < moves_edge[-1][2]:
        #         moves_edge = self.update_moves_list(moves_edge, (n1, n2, delta))
        #     n2 = tuple(g.out_edges(n2))[0][1]
        #
        # for n2 in unused_nodes:
        #     if n1 == n2:
        #         continue
        #     delta = self.__delta_replace_nodes(n1, n2)  # wyrzucenie wierzchołka i dodanie jednego spoza grafu
        #     if delta < moves_node[-1][2]:
        #         moves_node = self.update_moves_list(moves_node, (n1, n2, delta))
        #
        # # print("moves_node", moves_node)
        # # print("moves_edge", moves_edge)
        # return moves_edge, moves_node

    def streepest_edges_with_memory(self):
        g = self.g
        g_copy = self.g.copy()
        print("Random route length:", self._get_objective_function_value(g))

        unused_nodes = self.unused_nodes.copy()
        moves_edge, moves_node = self.generate_moves(unused_nodes)

        while True:
            if len(moves_edge) == 0 or len(moves_node) == 0:
                moves_edge, moves_node = self.generate_moves(unused_nodes)

            best_edge = moves_edge[0]
            best_node = moves_node[0]
            if best_edge[2] >= 0:
                break
            else:
                print("best_edge:",best_edge)
                self.__reorder_edge(best_edge[0], best_edge[1])

                for n in (best_edge[1], best_edge[0]):
                    n2 = tuple(g.out_edges(n))[0][1]
                    n1_prev = tuple(g.in_edges(n))[0][0]
                    best_for_new_node = (0, 0, 0)
                    while n2 != n1_prev:
                        delta = self.__delta_reorder_edge(n, n2)  # zamiana wierzchołków będących w grafie
                        if delta < best_for_new_node[2]:
                            best_for_new_node = (n, n2, delta)
                        n2 = tuple(g.out_edges(n2))[0][1]

                if best_for_new_node[2] < moves_edge[-1][2]:
                    moves_edge = self.update_moves_list(moves_edge, best_for_new_node)
                moves_edge = self.remove_from_list(moves_edge, best_edge[0])
            # else:
            #     print("best_node",best_node)
            #     print("unusd",unused_nodes)
            #     self.__replace_nodes(best_node[0], best_node[1])
            #     unused_nodes.remove(best_node[1])
            #     unused_nodes.append(best_node[0])
            #     print("unusd", unused_nodes)
            #
            #     best_for_new_node = (0, 0, 0)
            #     for n in unused_nodes:
            #         if n == best_node[1]:
            #             continue
            #
            #         delta = self.__delta_replace_nodes(best_node[1], n)
            #         if delta < best_for_new_node[2]:
            #             best_for_new_node = (best_node[1], n, delta)
            #
            #     if best_for_new_node[2] < moves_node[-1][2]:
            #         moves_node = self.update_moves_list(moves_node, best_for_new_node)
            #
            #     moves_node = self.remove_from_list(moves_node, best_node[0])
            #     moves_edge = self.remove_from_list(moves_edge, best_node[0])

        print("Route length:", self._get_objective_function_value(g))
        self.show_graph(g, "streepest_edges.png")
        self.g = g_copy
        return g, self._get_objective_function_value(g)

    def show_graph(self, graph, name: str):
        pos = dict(problem.node_coords)
        nx.draw_networkx(graph, pos=pos, font_size=6, node_size=50)
        plt.savefig(name)
        plt.clf()
        plt.show()


if __name__ == '__main__':
    # wczytywanie problemu
    problem = tsp.load('../data/kroA100.tsp')
    # problem = tsp.load('data/kroB100.tsp')

    # tworzenie grafu na podstawie problemu
    graph = problem.get_graph().to_directed()
    distance_matrix = nx.to_numpy_matrix(graph)
    # np.savetxt('distance_matrix.txt',distance_matrix,fmt='%.2f')

    # solver robi brr
    mean_values, mean_time, min_len, max_len = [0] * 5, [0] * 5, [(nx.DiGraph(), 90000)] * 5, [(nx.DiGraph(), 0)] * 5
    n = 100

    for j in range(0, n):
        print(j)
        lcs = LocalSearchSolver(graph, distance_matrix)
        func_list = [lcs.streepest_edges_with_memory]

        for i in range(0, len(func_list)):
            start_time = time.time()
            g, length = func_list[i]()
            mean_values[i] += length
            mean_time[i] += time.time() - start_time

            if length < min_len[i][1]:
                min_len[i] = (g, length)
            elif length > max_len[i][1]:
                max_len[i] = (g, length)

    for L in [mean_values, mean_time]:
        for i in range(0, len(L)):
            L[i] = round(L[i] / n, 2)

    # for m in min_len:
    #     LocalSearchSolver.show_graph(None, m[0], f"min_{m[1]}.png")
    #
    # for m in max_len:
    #     LocalSearchSolver.show_graph(None, m[0], f"max_{m[1]}.png")

    print(mean_values)
    print(mean_time)
    print(min_len)
    print(max_len)
