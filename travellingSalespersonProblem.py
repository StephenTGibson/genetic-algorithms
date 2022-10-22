import numpy as np
rng = np.random.default_rng()
import matplotlib
import matplotlib.pyplot as plt

from geneticAlgorithm import *

# to do:
# normalise graph costs
# scale penalty for invalid solutions in some way
# smart crossover process - increase chance to split at expensive paths

class TravellingSalespersonGA(GeneticAlgorithm):
    """
    A class to solve the Travelling Salesperson Problem using a genetic
    algorithm. Inherits multiple attributes and methods from Genetic
    Algorithm parent class

    Attributes
    ----------
    graph : np.array
        Weighted adjacency graph representing tsp (directed or undirected)
    numLocations : int
        Number of locations in the problem
    graphTotalCost
        Sum of all edges in problem graph
    meanEdgeCost
        Mean of all possible journeys, excludes the 0 cost of travelling to
        current location (not moving)

    Methods
    -------
    initPop()
        Randomly generates a population of valid solutions to the TSP posed
        by self.graph
    computeFitness(solution)
        Computes fitness value of solution, where a higher fitness indicates
        a solution that better meets the problem definition
    computeValid(solution)
        Returns a boolean indicating whether solution is valid
    computePenalty(solution)
        Returns a value proportional to how invalid the solution is
    performCrossover(parent1, parent2)
        Creates 2 child solutions from 2 parent solutions using genetic
        crossover
    performMutation(solution)
        Mutates solution by randomly selecting one of the locations and
        randomly switching it to another (or the same) location
    """

    # __doc__ = GeneticAlgorithm.__doc__# + __doc__
    def __init__(
            self,
            popSize,
            selectionMechanism,
            crossoverPoints,
            elitism,
            mutation,
            mutationRate,
            permitInvalidSolutions,
            graph,
            ):
        # store problem graph and various attributes of it
        self.graph = graph
        self.numLocations = self.graph.shape[0]
        self.graphTotalCost = np.sum(self.graph)
        self.meanEdgeCost = self.graphTotalCost /\
            (self.numLocations**2 - self.numLocations)
        # call parent class initialiser
        super().__init__(
            popSize,
            selectionMechanism,
            crossoverPoints,
            elitism,
            mutation,
            mutationRate,
            permitInvalidSolutions,
            )

    def initPop(self):
        """
        Randomly generates a population of valid solutions to the TSP posed
        by self.graph

        Returns
        -------
        pop : list
            Contains self.popSize solutions
        """

        pop = []
        for _ in range(self.popSize):
            # generate solution visiting each location in turn
            solution = np.arange(self.numLocations)
            # randomise order
            self.rng.shuffle(solution)
            # add extra location to end such that solution returns to origin
            solution = np.append(solution, solution[0])
            # store solution in pop list
            pop.append(solution)
        return pop

    def computeFitness(self, solution):
        """
        Computes fitness value of solution, where a higher fitness indicates
        a solution that better meets the problem definition

        Parameters
        ----------
        solution : np.array
            Contains integers representing locations which corresponds to a
            path through the problem graph

        Returns
        -------
        1/cost : float
            Fitness of solution, where higher value indicates higher quality
        """

        cost = 0
        # start at first location
        currentLocation = solution[0]
        # iterate over each subsequent location in solution
        for destination in solution[1:]:
            # add cost of journey from current to next location
            cost += self.graph[currentLocation, destination]
            # update current location
            currentLocation = destination
        # if solution is invalid add penalty to cost
        if not self.computeValid(solution):
            cost += self.computePenalty(solution)
        # return solution fitness as reciprocal of cost
        return 1/(cost**2)

    def computeValid(self, solution):
        """
        Returns a boolean indicating whether solution is valid

        Parameters
        ----------
        solution : np.array
            Contains integers representing locations which corresponds to a
            path through the problem graph

        Returns
        -------
        valid : boolean
            Indicates whether solution is valid
        """

        # valid if solution includes all locations, and first and final
        # locations are the same
        valid = (len(solution[:-1]) == len(np.unique(solution[:-1]))) &\
            (solution[0] == solution[-1])
        return valid

    def computePenalty(self, solution):
        """
        Returns a value proportional to how invalid the solution is

        Parameters
        ----------
        solution : np.array
            Contains integers representing locations which corresponds to a
            path through the problem graph

        Returns
        -------
        penalty : int
            Value of penalty to increase cost of solution by
        """

        # compute how many locations are not visited by the solution and
        # multiply by mean edge cost
        penalty = (solution[:-1].shape[0] -\
            np.unique(solution[:-1]).shape[0]) * self.meanEdgeCost
        # if solution does not return to starting location increase penalty
        # by mean edge cost
        if solution[0] != solution[-1]:
            penalty += self.meanEdgeCost
        return penalty

    def performCrossover(self, parent1, parent2):
        """
        Creates 2 child solutions from 2 parent solutions using genetic
        crossover. Number of crossover points can be constant or random,
        defined at initialisation

        Parameters
        ----------
        parent1 : np.array
            A solution from current population
        parent2 : np.array
            A solution from current population, cannot be the same as parent1

        Returns
        -------
        child1 : np.array
            A new solution
        child2 : np.array
            A new solution
        """

        # selects number of crossover points if random, from between 1 and 1
        # less than number of locations in solution
        if self.crossoverPoints == 'random':
            crossoverPoints = self.rng.choice(np.arange(1, parent1.shape[0]))
            numSections = crossoverPoints + 1
        else:
            numSections = self.crossoverPoints + 1
        # compute step size from number of sections into which solutions
        # are divided
        stepSize = int(parent1.shape[0] / numSections)

        if self.permitInvalidSolutions:
            # initialise both children as copies of parent1
            child1 = parent1.copy()
            child2 = parent1.copy()
            # iterate over sections of solutions
            for section in range(numSections):
                # if section number is even, replace section in child2 with
                # section from parent2
                if section % 2 == 0:
                    child2[section*stepSize:(section+1)*stepSize] =\
                        parent2[section*stepSize:(section+1)*stepSize]
                # if section number is odd, replace section in child1 with
                # section from parent2
                else:
                    child1[section*stepSize:(section+1)*stepSize] =\
                        parent2[section*stepSize:(section+1)*stepSize]

        else:
            # initialise children as invalid solutions filled with -1s
            child1 = np.ones_like(parent1) * -1
            child2 = child1.copy()
            # iterate over solution indices
            for locIdx in range(parent1.shape[0]):
                # for even numbered sections, use parent1 for child1
                if (locIdx // stepSize) % 2 == 0:
                    child1[locIdx] = parent1[locIdx]
                # for odd, use parent1 for child2
                else:
                    child2[locIdx] = parent1[locIdx]
            # fill gaps with remaining locations from parent2
            child1[np.arange(child1.shape[0])[child1==-1]] =\
                parent2[np.isin(parent2, child1, invert=True)]
            child2[np.arange(child2.shape[0])[child2==-1]] =\
                parent2[np.isin(parent2, child2, invert=True)]
        # append start location to end
        child1 = np.append(child1, child1[0])
        child2 = np.append(child2, child2[0])
        return child1, child2

    def performMutation(self, solution):
        """
        Mutates solution randomly

        Parameters
        ----------
        solution : np.array
            Contains integers representing locations which corresponds to a
            path through the problem graph

        Returns
        -------
        solution : np.array
            Contains integers representing locations which corresponds to a
            path through the problem graph
        """

        if self.permitInvalidSolutions:
            # select index of location to change
            idx = self.rng.choice(solution.shape[0])
            # randomly select new location
            solution[idx] = self.rng.choice(solution.shape[0]-1)

        else:
            # select 2 location indices to swap, ensure they are different
            idx1 = self.rng.choice(solution.shape[0])
            idx2 = self.rng.choice(solution.shape[0])
            while idx1 == idx2:
                idx2 = self.rng.choice(solution.shape[0])
            solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
        # append start location to end
        solution = np.append(solution, solution[0])
        return solution


def makeGraph(nodes, maxCost, directed=False):
    """
    Creates a weighted adjacency graph. Randomly assigns costs to each
    journey

    Parameters
    ----------
    nodes : int
        Number of nodes (locations) in graph
    maxCost :
        Maximum cost of a journey between 2 locations
    directed : boolean (default=False)
        Indicates if graph is directed

    Returns
    -------
    graph : np.array
        Weighted adjacency graph representing a travelling salesperson
        problem. Value at row x and col y represents cost of making journey
        from location x to location y
    """

    # initialise graph with 0 costs
    graph = np.zeros((nodes, nodes))
    # iterate over row indices
    for rowIdx in range(nodes):
        # update all destinations from rowIdx with a cost value selected
        # randomly from between 1 and maxCost
        graph[rowIdx, rowIdx+1:] = rng.choice(np.arange(1, maxCost+1),\
            nodes-(rowIdx+1))
    # if directed do the same for column indices
    if directed:
        # iterate over col indices
        for colIdx in range(nodes):
            # update all origins to colIdx with a cost value selected
            # randomly from between 1 and maxCost
            graph[colIdx+1:, colIdx] = rng.choice(np.arange(1, maxCost+1),\
                nodes-(colIdx+1))
    # if undirected then graph[row, col] = graph[col, row]
    else:
        graph += graph.T
    return graph

def makeRealGraph(numLocations, bounds):
    """
    Creates a graph representing 'real' locations: locations that have
    defined positions and are plottable.

    Parameters
    ----------
    numLocations : int
        Number of locations in graph
    bounds : np.array
        Contains upper limits for location coordinates. No limit for number
        of dimensions but dimensions other than 2 will cause problems
        plotting

    Returns
    -------
    graph : np.array
        Weighted adjacency graph representing a travelling salesperson
        problem. Value at row x and col y represents cost of making journey
        from location x to location y
    locationsArray : np.array
        For each location (axis=0), contains each dimension's coordinates
        (axis=1). 0 <= coordinate value < bound
    """

    locationsArray = np.zeros((numLocations, bounds.shape[0]))
    for dimIdx in range(bounds.shape[0]):
        # select coordinate values for all locations in each dimension
        locationsArray[:, dimIdx] = rng.uniform(low=0, high=bounds[dimIdx], size=numLocations)
    graph = np.zeros((numLocations, numLocations))
    for rowIdx in range(numLocations):
        for colIdx in range(rowIdx+1, numLocations):
            # compute euclidean distance between 2 locations
            graph[rowIdx, colIdx] =\
                np.linalg.norm(locationsArray[rowIdx] -\
                    locationsArray[colIdx])
    # as distances are based on real locations, graph is undirected
    graph += graph.T
    return graph, locationsArray
