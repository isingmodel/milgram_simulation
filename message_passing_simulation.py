import igraph as ig
import random
import numpy as np
from tqdm import tqdm


class SmallWorldSimulation:
    """
    A class that represents the Small World Simulation.

    Attributes:
        num_nodes (int): Number of nodes in the network.
        p_vanish (float): The probability of a message vanishing during transit.
        graph (igraph.Graph): The network graph.
        locations (dict): A dictionary containing node locations.
    """

    def __init__(self, num_nodes, p_vanish):
        """
        The constructor for the SmallWorldSimulation class.

        Args:
            num_nodes (int): Number of nodes in the network.
            p_vanish (float): The probability of a message vanishing during single transit.
        """

        self.num_nodes = num_nodes
        self.p_vanish = p_vanish
        self.graph, self.locations = self.create_network()

    def create_network(self):
        """
        Method to create the network of clusters of nodes.
 		To create a small world network, several random networks are first created, 
 		and then several connection lines are created between these random networks.
 		The location of each cluster is given uniformly on the lattice.

        Returns:
            g (igraph.Graph): The created network graph.
            locations (dict): A dictionary containing node locations.
        """

        # Set parameters
        num_clusters_side = int(np.sqrt(self.num_nodes)/10)  # Number of clusters along one side of the grid
        num_clusters = int(num_clusters_side * num_clusters_side)  # Total number of clusters
        nodes_per_cluster = self.num_nodes // num_clusters  # Nodes in each cluster
        self.num_nodes = num_clusters * nodes_per_cluster
        inter_cluster_edges = num_clusters * 50

        in_cluster_prob = min(150 / nodes_per_cluster, 1) #150~= dunbar number
        

        # Create separate clusters
        clusters = [ig.Graph.Erdos_Renyi(n=nodes_per_cluster,
                                         p=in_cluster_prob) for _ in range(num_clusters)]

        # Assign each cluster a location on a grid
        cluster_locations = {}
        for i in range(num_clusters_side):
            for j in range(num_clusters_side):
                cluster_id = i * num_clusters_side + j
                cluster_locations[cluster_id] = (i / (num_clusters_side - 1), j / (num_clusters_side - 1))

        # Assign locations to each node based on its cluster's location
        locations = {}
        for i, cluster in enumerate(clusters):
            for node in range(cluster.vcount()):
                node_global_id = i * nodes_per_cluster + node  # Global node ID across all clusters
                cluster_location = np.array(cluster_locations[i])
                node_location = cluster_location + np.random.normal(0, 0.02, 2)  # Nodes are close to their cluster
                locations[int(node_global_id)] = tuple(node_location)

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
        """
        Calculate the Euclidean distance between two nodes.

        Args:
            node1 (int): Node 1.
            node2 (int): Node 2.

        Returns:
            float: The Euclidean distance between node1 and node2.
        """
        loc1 = np.array(self.locations[node1])
        loc2 = np.array(self.locations[node2])
        return max(np.linalg.norm(loc1-loc2), 0.01)

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
