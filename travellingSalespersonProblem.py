import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from geneticAlgorithm import *

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
        return 1/cost

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

        # initialise both children as copies of parent1
        child1 = parent1.copy()
        child2 = parent1.copy()
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
        return child1, child2

    def performMutation(self, solution):
        """
        Mutates solution by randomly selecting one of the locations
        and randomly switching it to another (or the same) location

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

        # select index of location to change
        idx = self.rng.choice(solution.shape[0])
        # randomly select new location
        solution[idx] = self.rng.choice(solution.shape[0]-1)
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

    rng = np.random.default_rng()
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

if __name__ == '__main__':

    rng = np.random.default_rng()

    # create problem graph
    nodes = 22
    maxCost = 10
    directed = True
    graph = makeGraph(
        nodes,
        maxCost,
        directed
        )
    # initialise ga
    initialPopulation = 100
    selectionMechanism = 'Roulette'
    crossoverPoints = 'random'
    elitism = True
    mutation = True
    mutationRate = 0.15
    tsp = TravellingSalespersonGA(
        initialPopulation,
        selectionMechanism,
        crossoverPoints,
        elitism,
        mutation,
        mutationRate,
        graph,
        )
    # perform evolution
    generations = 1000
    earlyStopGenerations = 400
    tsp.evolve(
        generations,
        earlyStopGenerations,
        )
    # random benchmark
    rand = TravellingSalespersonGA(
        initialPopulation,
        selectionMechanism,
        crossoverPoints,
        elitism,
        mutation,
        mutationRate,
        graph,
        )
    # perform random
    rand.randomBenchmark(generations)

    # results
    factorial = 1
    current = 1
    while current <= nodes:
        factorial = current * factorial
        current += 1
    print(f'Permutations: {(factorial):.1e}')
    print(f'Max possible solutions evaluated: {(initialPopulation*generations):.1e}')

    # print(tsp.validArrHistory[0])
    # print(tsp.validArrHistory[-1])

    # print(tsp.pop[np.argmax(tsp.fitArr)])

    benchmarkSolution = np.append(np.arange(nodes), 0)
    benchmarkFit = tsp.computeFitness(benchmarkSolution)

    # plots
    fig, axs = plt.subplots(2, 2, figsize=(10,6), sharey=True)

    cmap = matplotlib.colors.ListedColormap(['red', 'blue'])

    tspRows = np.ones_like(tsp.fitArrHistory)
    for row in range(tsp.fitArrHistory.shape[0]):
        tspRows[row] *= row
    axs[0, 0].scatter(
        tspRows.flatten(),
        tsp.fitArrHistory.flatten(),
        s=0.3,
        c=tsp.validArrHistory.flatten(),
        cmap=cmap,
    )

    randRows = np.ones_like(rand.fitArrHistory)
    for row in range(rand.fitArrHistory.shape[0]):
        randRows[row] *= row
    axs[0, 1].scatter(
        randRows.flatten(),
        rand.fitArrHistory.flatten(),
        s=0.3,
        c=rand.validArrHistory.flatten(),
        cmap=cmap,
    )
    # axs[0].legend()

    axs[1, 0].plot(
        range(tsp.history.shape[0]),
        tsp.history[:,0],
        label='Max')
    axs[1, 0].plot(
        range(tsp.history.shape[0]),
        tsp.history[:,1],
        label='Mean')
    axs[1, 0].hlines(
        benchmarkFit,
        0,
        tsp.history.shape[0],
        linestyles='dashed',
        colors='k',
        label='Benchmark')

    # # # = axs[1].twinx()
    # axs[1, 1].plot(
    #     range(tsp.history.shape[0]),
    #     tsp.history[:,2],
    #     label='Unique solutions %',
    #     # color='red'
    #     )

    # axs[1, 1].plot(
    #     range(tsp.history.shape[0]),
    #     tsp.history[:,3],
    #     label='Valid solutions %',
    #     # color='red'
    #     )

    # # ax1.plot(x, y1, 'g-')
    # # ax2.plot(x, y2, 'b-')

    # # ax1.set_xlabel('X data')
    # # ax1.set_ylabel('Y1 data', color='g')
    # # ax2.set_ylabel('Y2 data', color='b')

    # axs[1].legend()
    # axs[2].legend()#loc=(1.01,0.5))
    plt.show()
