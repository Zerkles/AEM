import tsplib95 as tsp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

class Solver:

    def __init__(self,graph,distance_matrix):
        self.graph = graph
        self.graph.clear_edges()
        self.distance_matrix=distance_matrix

    def find_nn(self,node:int, nodes_list:list):
        distance_matrix = self.distance_matrix
        nn= nodes_list[0]
        min_distance = np.min(distance_matrix[nn-1,node-1])
        
        for n in nodes_list[1:]:
            distance = np.min(distance_matrix[n-1,node-1])
        
            if distance < min_distance:
                nn = n
                min_distance = distance
        
        return nn

    def find_shortest_loop(self,node,edges):
        distance_matrix = self.distance_matrix
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

    def calculate_regret(self,node,edges):
        distance_matrix=self.distance_matrix
        nn1,nn2,distance1= self.find_shortest_loop(node,edges)
        edges.remove((nn1,nn2))

        nn1,nn2,distance2= self.find_shortest_loop(node,edges)
        edges.remove((nn1,nn2))

        nn1,nn2,distance3= self.find_shortest_loop(node,edges)

        return distance1-distance2 + distance1-distance3

    @staticmethod
    def calculate_length(graph,distance_matrix):
        g,dm = graph, distance_matrix
        sum = 0
        for e in g.edges():
            u,v = e
            sum+=dm[u-1,v-1]

        return sum

    def greedy_nn(self):
        g=self.graph.copy()

        start_node = random.choice(list(g.nodes))
        node = start_node

        unused_nodes = list(g.nodes)
        unused_nodes.remove(node)
        
        while g.number_of_edges() < int(g.number_of_nodes()*0.5):
            nn = self.find_nn(node,unused_nodes)
            g.add_edge(node,nn)
            unused_nodes.remove(nn)
            node = nn

        g.add_edge(node,start_node)
        return g, Solver.calculate_length(g,distance_matrix)


    def greedy_cycle(self):
        g= self.graph.copy()

        start_node = random.choice(list(g.nodes))
        node = start_node

        unused_nodes = list(g.nodes)
        unused_nodes.remove(node)
        
        #creating first cycle // first edge
        nearest_neighbor = self.find_nn(node,unused_nodes)
        g.add_edge(node,nearest_neighbor)
        unused_nodes.remove(nearest_neighbor)
        node = nearest_neighbor
        
        # second edge
        nearest_neighbor = self.find_nn(node,unused_nodes)
        g.add_edge(node,nearest_neighbor)
        unused_nodes.remove(nearest_neighbor)
        node = nearest_neighbor

        # third edge to start point
        g.add_edge(node,start_node)

        while g.number_of_edges() != int(g.number_of_nodes()*0.5):
            shortest_loop_node = unused_nodes[0]
            _,_,shortest_loop = self.find_shortest_loop(shortest_loop_node,list(g.edges()))
            
            for n in unused_nodes[1:]: 
                _,_,loop = self.find_shortest_loop(n,list(g.edges()))
                
                if loop<shortest_loop:
                    shortest_loop_node = n
                    shortest_loop = loop

            nn1,nn2,_= self.find_shortest_loop(shortest_loop_node,list(g.edges()))
            
            g.remove_edge(nn1,nn2)
            g.add_edge(shortest_loop_node,nn1)
            g.add_edge(shortest_loop_node,nn2)

            unused_nodes.remove(shortest_loop_node)
        return g, Solver.calculate_length(g,distance_matrix)

    def regret_heuristic(self):
        g=self.graph.copy()
        
        start_node = random.choice(list(g.nodes))
        node = start_node

        unused_nodes = list(g.nodes)
        unused_nodes.remove(node)
        
        #creating first cycle // first edge
        nearest_neighbor = self.find_nn(node,unused_nodes)
        g.add_edge(node,nearest_neighbor)
        unused_nodes.remove(nearest_neighbor)
        node = nearest_neighbor
        
        # second edge
        nearest_neighbor = self.find_nn(node,unused_nodes)
        g.add_edge(node,nearest_neighbor)
        unused_nodes.remove(nearest_neighbor)
        node = nearest_neighbor

        # third edge to start point
        g.add_edge(node,start_node)

        while g.number_of_edges() != int(g.number_of_nodes()*0.5):

            max_regret_node = unused_nodes[0]
            max_regret = self.calculate_regret(max_regret_node,list(g.edges()))
            for n in unused_nodes[1:]: 
                regret = self.calculate_regret(n,list(g.edges()))
                
                if regret < max_regret:
                    max_regret_node = n
                    max_regret = regret

            nn1,nn2,_= self.find_shortest_loop(max_regret_node,list(g.edges()))
            
            g.remove_edge(nn1,nn2)
            g.add_edge(max_regret_node,nn1)
            g.add_edge(max_regret_node,nn2)

            unused_nodes.remove(max_regret_node)
        return g, Solver.calculate_length(g,distance_matrix)

    @staticmethod
    def show_graph(graph,name:str):
        pos = dict(problem.node_coords)
        nx.draw_networkx(graph, pos=pos, font_size=6, node_size=50)
        plt.savefig(name)
        plt.clf()
        plt.show()
    
    @staticmethod
    def show_statistics(graph,distance_matrix):
        alg_count = 3
        mean_values,max_len,min_len = [0]*alg_count,[0]*alg_count,[(None,0)]*alg_count
        n=50

        for j in range(0,n):
            print(j)
            s = Solver(graph,distance_matrix)
            func_list = [s.greedy_nn,s.greedy_cycle,s.regret_heuristic]
            
            for i in range(0,len(func_list)):
                g,length = func_list[i]()
                mean_values[i]+= length
                print(length)

                if length<min_len[i][1] or min_len[i][0] == None:
                    min_len[i]=(g,length)
                elif length>max_len[i]:
                    max_len[i]=length

            
        
        for i in range(0,len(mean_values)):
            mean_values[i] = round(mean_values[i]/n,2)
        
        for m in min_len:
            Solver.show_graph(m[0],f"min_{m[1]}.png")

        print("Mean values:",mean_values)
        print("Shortest length:",min_len)
        print("Longest length",max_len)

if __name__ == '__main__':
    #wczytywanie problemu
    problem = tsp.load('data/kroB100.tsp')

    #tworzenie grafu na podstawie problemu
    graph = problem.get_graph()
    distance_matrix = nx.to_numpy_matrix(graph)
    np.savetxt('distance_matrix.txt',distance_matrix,fmt='%.2f')

    #rozwiązywanie problemu
    Solver.show_statistics(graph,distance_matrix)

    
