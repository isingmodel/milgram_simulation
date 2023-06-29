import igraph as ig
import numpy as np
import random
from tqdm import tqdm


def create_geographical_smallworld_network(n_nodes, k_nearest, rewiring_prob):
    #n_nodes, k_nearest, rewiring_prob = 900, 70, 0.03

    # Ensure a perfect square number of nodes for a 2D lattice
    # assert np.sqrt(n_nodes) == int(np.sqrt(n_nodes)), "n_nodes should be a perfect square for a 2D lattice"

    side_length = int(np.sqrt(n_nodes))

    # Initialize the graph with n_nodes vertices
    g = ig.Graph(n=n_nodes)
    g = g.as_undirected()

    # Initialize the list of node coordinates
    node_coordinates = []

    # Create the nodes and arrange them in a 2D lattice
    for i in range(side_length):
        for j in range(side_length):
            x = i / (side_length - 1) # Normalize the coordinates to [0,1]
            y = j / (side_length - 1)
            node_coordinates.append((x, y))

    # Add edges between each node and its k nearest neighbors
    # Define the distance metric function
    distance = lambda x, y : np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    # Iterate over each node
    for idx, node in enumerate(node_coordinates):
        # Calculate distances to all other nodes
        distances = [distance(node, other_node) for other_node in node_coordinates]
        # Get indices of k nearest neighbors
        indices = np.argsort(distances)[1:k_nearest+1] # excluding the node itself
        # Add edges to k nearest neighbors
        g.add_edges([(idx, neighbor) for neighbor in indices])

    # Perform rewiring
    for edge in g.es:
        try:
            # With a probability of rewiring_prob, rewire the edge
            if random.random() < rewiring_prob:
                # Select a new node randomly to maintain the total number of edges
                new_node = random.randint(0, n_nodes-1)
                # Perform rewiring while avoiding self-loops and edge duplication
                if new_node not in [edge.source, edge.target] and g.get_eid(edge.source, new_node, directed=False, error=False) == -1:
                    # Delete the existing edge
                    g.delete_edges(edge)
                    # Add a new edge
                    g.add_edge(edge.source, new_node)
        except:
            pass
    # Return the node coordinates and the graph
    return node_coordinates, g


class SmallWorldSimulation:
    """
    A class that represents the Small World Simulation.

    Attributes:
        num_nodes (int): Number of nodes in the network.
        p_vanish (float): The probability of a message vanishing during transit.
        graph (igraph.Graph): The network graph.
        locations (dict): A dictionary containing node locations.
    """

    def __init__(self, num_nodes_city, num_city, k_nearest, rewiring_prob, p_vanish):


        self.num_nodes_city = num_nodes_city
        self.num_city = num_city
        self.num_nodes = self.num_nodes_city * self.num_city * self.num_city
        self.p_vanish = p_vanish
        self.k_nearest = k_nearest
        self.rewiring_prob = rewiring_prob
        self.graph, self.locations = self.create_network()


    def create_network(self):
        # Set parameters
        num_clusters_side = self.num_city  # Number of clusters along one side of the grid
        num_clusters = int(num_clusters_side * num_clusters_side)  # Total number of clusters
        nodes_per_cluster = self.num_nodes_city  # Nodes in each cluster
        inter_cluster_edges = num_clusters * 50

#         in_cluster_prob = min(150 / nodes_per_cluster, 1) #150~= dunbar number
        

        # Create separate clusters
#         clusters = [ig.Graph.Erdos_Renyi(n=nodes_per_cluster,
#                                          p=in_cluster_prob) for _ in range(num_clusters)]
        
        

        # Assign each cluster a location on a grid
        locations_list = []
        clusters = []
        for idx in range(num_clusters):
            loc_in_cluster, graph = create_geographical_smallworld_network(nodes_per_cluster,
                                                                           self.k_nearest,
                                                                           self.rewiring_prob)
            clusters.append(graph)
            
            i = idx // num_clusters_side
            j = idx % num_clusters_side
            loc = [(x/50 + (i / num_clusters_side), y/50 + (j/ num_clusters_side)) for x, y in loc_in_cluster]
            locations_list += loc
            
        # Assign locations to each node based on its cluster's location
        locations = {}
        for idx in range(len(locations_list)):
            locations[idx] = locations_list[idx]


        # Combine all clusters into a single graph
        g = clusters[0]
        for i in range(1, num_clusters):
            g = g + clusters[i]

        # Add inter-cluster edges to create a small-world effect
        for _ in tqdm(range(inter_cluster_edges)):
            # Choose two clusters randomly
            cluster1, cluster2 = np.random.choice(num_clusters, 2, replace=False)
            # Choose a node from each cluster
            node1 = np.random.choice(range(cluster1 * nodes_per_cluster, (cluster1 + 1) * nodes_per_cluster))
            node2 = np.random.choice(range(cluster2 * nodes_per_cluster, (cluster2 + 1) * nodes_per_cluster))
            # Add an edge between the chosen nodes
            g.add_edge(node1, node2)

        return g, locations


    def calculate_distance(self, node1, node2):
        loc1 = np.array(self.locations[node1])
        loc2 = np.array(self.locations[node2])
        return max(np.linalg.norm(loc1-loc2), 0.001)

    def simulate(self):
        """
        Simulate the routing of a message from source to target(final destination) in the network.

        Returns:
            steps (int): The number of steps taken for the message to reach the target.
            -1: If the message vanishes during transit.
            -2: If there are no neighbors to route the message to.
        """
        ...
        source = random.randint(0, self.num_nodes - 1)
        target = random.randint(0, self.num_nodes - 1)
        while target == source:
            target = random.randint(0, self.num_nodes - 1)

        current_node = source
        prev_node = None
        steps = 0

        while True:
            # Mail vanishes with probability p_vanish
            if random.random() < self.p_vanish:
                return -1  # Mail vanished

            # Check if mail reached target
            if current_node == target:
                return steps  # Mail reached target
            
            neighbors = self.graph.neighbors(current_node)
                
            # Can't send mail back to the node that sent it to us
            if prev_node is not None and prev_node in neighbors:
                neighbors.remove(prev_node)
                
            if len(neighbors) == 0:
                return -2

            # Choose next node based on distance to target
            distances = [self.calculate_distance(neighbor, target) for neighbor in neighbors]
            probabilities = [1./dist for dist in distances]  # closer nodes have higher probability
            weights = [p/sum(probabilities) for p in probabilities]  # normalize probabilities

            prev_node = int(current_node)
            current_node = int(np.random.choice(neighbors, p=weights))

            
            steps += 1