import tsplib95 as tsp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

class LocalSearchSolver:
    def __init__(self, g:nx.DiGraph,distance_matrix:np.matrix):
        self.g = g
        self.dm = distance_matrix
    
    def __find_nn(node:int, nodes_list:list, distance_matrix:np.matrix):
        nn= nodes_list[0]
        min_distance = np.min(distance_matrix[nn-1,node-1])
        
        for n in nodes_list[1:]:
            distance = np.min(distance_matrix[n-1,node-1])
        
            if distance < min_distance:
                nn = n
                min_distance = distance
        
        return nn

    def __find_shortest_loop(node,edges,distance_matrix):
        n1_min,n2_min = edges[0]
        min_loop = distance_matrix[node-1,n1_min-1] + distance_matrix[node-1,n2_min-1] - distance_matrix[n1_min-1,n2_min-1]
        
        for e in edges[1:]:
            n1,n2 = e
            loop = distance_matrix[node-1,n1-1]+distance_matrix[node-1,n2-1] - distance_matrix[n1-1,n2-1]
            if loop<min_loop:
                min_loop=loop
                n1_min = n1
                n2_min = n2

        return n1_min, n2_min, min_loop

    def __replace_nodes(self, n1,n2):
        g = self.g
        print(n1,n2)
        print(n1,tuple(g.in_edges(n1)),tuple(g.out_edges(n1)))
        n1_prev,n1_next = tuple(g.in_edges(n1))[0][0],tuple(g.out_edges(n1))[0][1]

        g.remove_edge(n1_prev,n1)
        g.remove_edge(n1,n1_next)

        g.add_edge(n1_prev,n2)
        g.add_edge(n2,n1_next)

    def __reorder_nodes(self,n1,n2):
        g = self.g
        n1_prev,n1_next = tuple(g.in_edges(n1))[0][0],tuple(g.out_edges(n1))[0][1]
        n2_prev,n2_next = tuple(g.in_edges(n2))[0][0],tuple(g.out_edges(n2))[0][1]

        g.remove_edge(n1_prev,n1)
        g.remove_edge(n1,n1_next)
        g.remove_edge(n2_prev,n2)
        g.remove_edge(n2,n2_next)

        g.add_edge(n1_prev,n2)
        g.add_edge(n2,n1_next)
        g.add_edge(n2_prev,n1)
        g.add_edge(n1,n2_next)

    def __exchange_route_edges(e1,e2):
        pass

    def __delta_reorder_nodes(self, n1,n2):
        g,dm = self.g,self.dm

        n1_prev,n1_next = tuple(g.in_edges(n1))[0][0],tuple(g.out_edges(n1))[0][1]
        n2_prev,n2_next = tuple(g.in_edges(n2))[0][0],tuple(g.out_edges(n2))[0][1]

        old_distance = dm[n1_prev-1,n1-1] +dm[n2_prev-1,n2-1] + dm[n2-1,n2_next-1] + dm[n1-1,n1_next-1]
        new_distance = dm[n1_prev-1,n2-1] +dm[n2_prev-1,n1-1] + dm[n2-1,n1_next-1] + dm[n1-1,n2_next-1]

        return old_distance - new_distance

    def __delta_replace_nodes(self, n1,n2): # n1 jest w ścieżce, n2 nie
        g,dm = self.g,self.dm

        n1_prev,n1_next = tuple(g.in_edges(n1))[0][0],tuple(g.out_edges(n1))[0][1]
        
        old_distance = dm[n1_prev-1,n1-1] +dm[n1-1,n1_next-1]
        new_distance = dm[n1_prev-1,n2-1] +dm[n2-1,n1_next-1]

        return old_distance - new_distance

    def __get_objective_function_value(self):
        g,dm = self.g,self.dm
        sum = 0
        for e in g.edges():
            u,v = e
            sum+=dm[u-1,v-1]

        return sum

    def create_random_route(self):
        g,dm = self.g, self.dm
        
        g.clear_edges()
        unused_nodes = list(g.nodes)
        start_node = node = random.choice(unused_nodes)
        unused_nodes.remove(start_node)
        
        while g.number_of_edges() != int(g.number_of_nodes()*0.5)-1:
            new_node = random.choice(unused_nodes)
            g.add_edge(node, new_node)
            unused_nodes.remove(new_node)
            node = new_node
        g.add_edge(node, start_node)

        return unused_nodes

    def random_alg():
        pass

    def streepest_nodes(self):
        g,dm = self.g,self.dm
        unused_nodes = self.create_random_route()
        used_nodes = list(g.nodes - unused_nodes)

        route_len = self.__get_objective_function_value()
        print("Route length:", route_len)

        best_move1, best_move2 = (0,0,-1),(0,0,-1)
        while best_move1[2]<0 or best_move2[2]<0:
            # format: (wierzchołek1,wierzchołek2,wartość delty funkcji celu)
            best_move1, best_move2 = (0,0,0),(0,0,0)
            for n1 in used_nodes:
                for n2 in used_nodes:
                    if n1==n2:
                        continue
                    # zamiana wierzchołków będącychw grafie
                    delta = self.__delta_reorder_nodes(n1,n2)
                    if best_move1[2]>delta:
                        best_move1 = (n1,n2,delta)
                
                for n2 in unused_nodes:
                    if n1==n2:
                        continue
                    # wyrzucenie wierzchołka i dodanie jednego spoza grafu
                    delta = self.__delta_replace_nodes(n1,n2)
                    if best_move2[2]>delta:
                        best_move2 = (n1,n2,delta)

                

            if best_move1 < best_move2:
                self.__reorder_nodes(best_move1[0],best_move1[1])
            else:
                self.__replace_nodes(best_move2[0],best_move2[1])
                used_nodes.remove(best_move2[0])
                unused_nodes.remove(best_move2[1])
                used_nodes.append(best_move2[1])
                unused_nodes.append(best_move2[0])

        route_len = self.__get_objective_function_value()
        print("Route length:",route_len)    
        


    def streepest_edges():
        pass

    def greedy_nodes():
        pass

    def greedy_edges():
        pass

    def show_graph(self):
        pos = dict(problem.node_coords)
        nx.draw_networkx(graph, pos=pos, font_size=6, node_size=50)
        plt.savefig("graph.png")
        plt.show()

if __name__ == '__main__':
    #wczytywanie problemu
    problem = tsp.load('data/kroA100.tsp')

    #tworzenie grafu na podstawie problemu
    graph = problem.get_graph().to_directed()
    distance_matrix = nx.to_numpy_matrix(graph)
    np.savetxt('distance_matrix.txt',distance_matrix,fmt='%.2f')

    # solver robi brr
    lcs = LocalSearchSolver(graph,distance_matrix)
    lcs.streepest_nodes()
    lcs.show_graph()
    
