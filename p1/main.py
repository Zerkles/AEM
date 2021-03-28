import tsplib95 as tsp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

def greedy_nn(g: nx.Graph, distance_matrix:np.matrix):
    g.clear_edges()
    
    start_node = random.choice(list(g.nodes))-1
    max_distance = distance_matrix.max()+1
    node = start_node

    print("Start node:", start_node, "Max distance:", max_distance)
    
    while g.number_of_edges() != int(g.number_of_nodes()*0.5):
        min_distance = np.min(distance_matrix[node][np.nonzero(distance_matrix[node])])
        print("Smallest distance:",min_distance)
        
        nearest_neighbor = np.where(distance_matrix[node] == min_distance)[1][0]
        print("Nearest Neighbor:",nearest_neighbor)
        
        g.add_edge(node+1,nearest_neighbor+1)
        distance_matrix[:,node] = max_distance

        node = nearest_neighbor

    g.add_edge(node+1,start_node+1)
    return g

def find_nn(node:int, nodes_list:list, distance_matrix:np.matrix):
    nn= nodes_list[0]
    min_distance = np.min(distance_matrix[nn-1,node-1])
    
    for n in nodes_list[1:]:
        distance = np.min(distance_matrix[n-1,node-1])
       
        if distance < min_distance:
            nn = n
            min_distance = distance
    
    return nn

def find_shortest_loop(node,edges,distance_matrix):
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

def greedy_cycle(g: nx.Graph, distance_matrix:np.matrix):
    g.clear_edges()
    
    start_node = random.choice(list(g.nodes))
    node = start_node

    unused_nodes = list(g.nodes)
    unused_nodes.remove(node)
    print("Start node:", start_node)
    
    #creating first cycle // first edge
    nearest_neighbor = find_nn(node,unused_nodes,distance_matrix)
    g.add_edge(node,nearest_neighbor)
    unused_nodes.remove(nearest_neighbor)
    node = nearest_neighbor
    
    # second edge
    nearest_neighbor = find_nn(node,unused_nodes,distance_matrix)
    g.add_edge(node,nearest_neighbor)
    unused_nodes.remove(nearest_neighbor)
    node = nearest_neighbor

    # third edge to start point
    g.add_edge(node,start_node)

    while g.number_of_edges() != int(g.number_of_nodes()*0.5):

        shortest_loop_node = unused_nodes[0]
        _,_,shortest_loop = find_shortest_loop(shortest_loop_node,list(g.edges()),distance_matrix)
        
        for n in unused_nodes[1:]: 
            _,_,loop = find_shortest_loop(n,list(g.edges()),distance_matrix)
            
            if loop<shortest_loop:
                shortest_loop_node = n
                shortest_loop = loop

        nn1,nn2,_= find_shortest_loop(shortest_loop_node,list(g.edges()),distance_matrix)
        
        g.remove_edge(nn1,nn2)
        g.add_edge(shortest_loop_node,nn1)
        g.add_edge(shortest_loop_node,nn2)

        unused_nodes.remove(shortest_loop_node)
    return g

def calculate_regret(node,edges,distance_matrix):
    nn1,nn2,distance1= find_shortest_loop(node,edges,distance_matrix)
    edges.remove((nn1,nn2))

    nn1,nn2,distance2= find_shortest_loop(node,edges,distance_matrix)
    edges.remove((nn1,nn2))

    nn1,nn2,distance3= find_shortest_loop(node,edges,distance_matrix)

    return distance1-distance2 + distance1-distance3

def regret_heuristic(g: nx.Graph, distance_matrix:np.matrix):
    g.clear_edges()
    
    start_node = random.choice(list(g.nodes))
    node = start_node

    unused_nodes = list(g.nodes)
    unused_nodes.remove(node)

    print("Start node:", start_node)
    
    #creating first cycle // first edge
    nearest_neighbor = find_nn(node,unused_nodes,distance_matrix)
    g.add_edge(node,nearest_neighbor)
    unused_nodes.remove(nearest_neighbor)
    node = nearest_neighbor
    
    # second edge
    nearest_neighbor = find_nn(node,unused_nodes,distance_matrix)
    g.add_edge(node,nearest_neighbor)
    unused_nodes.remove(nearest_neighbor)
    node = nearest_neighbor

    # third edge to start point
    g.add_edge(node,start_node)

    while g.number_of_edges() != int(g.number_of_nodes()*0.5):

        max_regret_node = unused_nodes[0]
        max_regret = calculate_regret(max_regret_node,list(g.edges()),distance_matrix)
        for n in unused_nodes[1:]: 
            regret = calculate_regret(n,list(g.edges()),distance_matrix)
            
            if regret < max_regret:
                max_regret_node = n
                max_regret = regret

        nn1,nn2,_= find_shortest_loop(max_regret_node,list(g.edges()),distance_matrix)
        
        g.remove_edge(nn1,nn2)
        g.add_edge(max_regret_node,nn1)
        g.add_edge(max_regret_node,nn2)

        unused_nodes.remove(max_regret_node)
    return g

if __name__ == '__main__':
    #wczytywanie problemu
    problem = tsp.load('data/kroA100.tsp')

    #tworzenie grafu na podstawie problemu
    graph = problem.get_graph()
    distance_matrix = nx.to_numpy_matrix(graph)
    np.savetxt('distance_matrix.txt',distance_matrix,fmt='%.2f')

    #rozwiązywanie problemu
    # graph = greedy_nn(graph,distance_matrix)
    # graph = greedy_cycle(graph,distance_matrix)
    graph = regret_heuristic(graph,distance_matrix)

    #rysowanie grafu
    pos = dict(problem.node_coords)
    nx.draw_networkx(graph, pos=pos, font_size=6, node_size=50)
    plt.savefig("graph.png")