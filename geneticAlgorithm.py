import numpy as np

class GeneticAlgorithm():
    """
    A class to formulate a genetic algorithm

    Attributes
    ----------
    popSize : int
        Number of members in each generation (constant)
    selectionMechanism : str
        Mechanism by which parents are selected
        Implemented: 'Roulette'
    crossoverPoints : int or str ('random')
        Number of crossover points when performing crossover. If 'random'
        then select uniformly randomly number of points between 1 and 1 less
        than solution length
    elitism : boolean
        Whether elitism is used when generating subsequent populations
    mutation : boolean
        Whether mutation  is used when generating subsequent populations
    nonMutationRate : float [0 - 1]
        Chance of mutation not occuring for each population member
    rng : NumPy random generator
        Used to generate random selections
    selectionMechanismDict : dict
        Links selection methods to input string
    pop : list
        Stores current generation's population of solutions
    history : np.array
        Stores max fitness, mean fitness, % unique solutions, % valid
        solutions for each generation
    fitArrHistory : np.array
        Stores fitness of every solution for each generation
    validArrHistory : np.array
        Stores validity of every solution for each generation

    Methods
    -------
    recordHistory()
        Stores information about current generation
    evolve(generations, earlyStop)
        Runs GA for some number of generations
    createNextGen()
        Creates next generation from current population
    computeFitnessArray()
        Creates array of fitness values for current population
    computeValidArray()
        Creates validity array for current population
    rouletteSelect(standardisedFitArr)
        Selects 1 member of the population using roulette selection
    randomBenchmark(generations)
        Repeatedly generates random solutions to benchmark GA performance
    initPop()
        Problem specific, to be overwritten in child class
    computeFitness()
        Problem specific, to be overwritten in child class
    computeValid()
        Problem specific, to be overwritten in child class
    performCrossover()
        Problem specific, to be overwritten in child class
    performMutation()
        Problem specific, to be overwritten in child class
    """

    def __init__(self, popSize, selectionMechanism, crossoverPoints,
    elitism, mutation, mutationRate, permitInvalidSolutions,):
        """
        Initialises 1st generation population and arrays to store history

        Parameters
        ----------
        popSize : int
            Number of members in each generation (constant)
        selectionMechanism : str
            Mechanism by which parents are selected
            Implemented: 'Roulette'
        crossoverPoints : int or str ('random')
            Number of crossover points when performing crossover. If 'random'
            then select uniformly randomly number of points between 1 and 1
            less than solution length
        elitism : boolean
            Whether elitism is used when generating subsequent populations
        mutation : boolean
            Whether mutation  is used when generating subsequent populations
        mutationRate : float [0 - 1]
            Chance of mutation occuring for each population member
        permitInvalidSolutions : boolean
            Indicates whether invalid solutions can exist in population
        """

        self.popSize = popSize
        self.selectionMechanism = selectionMechanism
        self.crossoverPoints = crossoverPoints
        self.elitism = elitism
        self.mutation = mutation
        self.permitInvalidSolutions = permitInvalidSolutions
        # need to enforce that mutation rate is between 0 and 1
        self.nonMutationRate = 1. - mutationRate
        # used for generating random choices
        self.rng = np.random.default_rng()
        # used to identify which selection mechanism to use
        self.selectionMechanismDict = {
            'Roulette': self.rouletteSelect,
            'Other': None,
        }
        # initialise first generation population
        self.pop = self.initPop()
        # initialise arrays to store GA history
        self.history = np.empty((0,4))
        self.fitArrHistory = np.empty((0, self.popSize))
        self.validArrHistory = np.empty((0, self.popSize))
        return

    def recordHistory(self):
        """
        Stores details of current population
        """

        # add fitness and validity arrays to histories
        self.fitArrHistory = np.append(
            self.fitArrHistory,
            np.array([self.fitArr]), axis=0)
        self.validArrHistory = np.append(
            self.validArrHistory,
            np.array([self.validArr]), axis=0)
        # add general history to array
        self.history = np.append(
            self.history,
            np.array([
                [np.max(self.fitArr[self.validArr])],
                [np.mean(self.fitArr[self.validArr])],
                # % of solutions that are unique
                [(np.unique(np.array(self.pop),
                        axis=0).shape[0])/self.popSize],
                # % of solutions that are valid
                [(np.sum(self.validArr))/self.popSize],
                ]).T,
                axis=0)
        return

    def evolve(self, generations, earlyStop):
        """
        Runs GA for some number of generations

        Parameters
        ----------
        generations : int
            Number of generations to compute
        earlyStop : int
            Number of generations after which algorithm is halted if no
            improvement is seen in maximum fitness
        """

        # iterate over number of generations
        for gen in range(generations):
            # create subsequent generation
            self.pop = self.createNextGen()
            self.recordHistory()
            # check if max fitness has increased since earlyStop generations
            if gen > earlyStop and\
                np.unique(self.history[-1*earlyStop:,0]).shape[0] == 1:
                break
        # evaluate final generation and store results
        self.fitArr = self.computeFitnessArray()
        self.validArr = self.computeValidArray()
        self.recordHistory()
        return

    def createNextGen(self):
        """
        Create next generation from current population

        Returns
        -------
        newPop : list
            Contains newly generated population of solutions
        """

        self.fitArr = self.computeFitnessArray()
        # standardise fitness values
        standardisedFitArr = self.fitArr/np.sum(self.fitArr)
        self.validArr = self.computeValidArray()

        newPop = []
        # loop until new population is same size as current
        while len(newPop) < self.popSize:
            # select 2 parents to perform crossover
            parentIdx1 = self.selectionMechanismDict[
                self.selectionMechanism](standardisedFitArr)
            parentIdx2 = self.selectionMechanismDict[
                self.selectionMechanism](standardisedFitArr)
            # enforce parents must be different
            while (self.pop[parentIdx1] == self.pop[parentIdx2]).all():
                parentIdx2 = self.selectionMechanismDict[
                    self.selectionMechanism](standardisedFitArr)
            # generate 2 children and add them to new population
            newPop.extend(self.performCrossover(
                self.pop[parentIdx1][:-1],
                self.pop[parentIdx2][:-1],
                ))

        if self.mutation:
            # iterate over idx of each solution
            for solIdx in range(len(newPop)):
                # generate random value [0, 1) and compare to mutation rate
                rand = self.rng.uniform()
                if rand > self.nonMutationRate:
                    # remove, perform mutation and re-append
                    solution = newPop.pop(solIdx)[:-1]
                    mutatedSolution = self.performMutation(solution)
                    newPop.append(mutatedSolution)

        if self.elitism:
            # randomly remove 1 solution
            newPop.pop(self.rng.choice(len(newPop)))
            # append highest fitness valid solution from current population
            newPop.append(np.array(self.pop)[self.validArr]
                            [np.argmax(self.fitArr[self.validArr])])
        return newPop

    def computeFitnessArray(self):
        """
        Creates array of fitness values for current population

        Returns
        -------
        fitArr : np.array
            Contains fitness value of each solution
        """

        return np.array([self.computeFitness(solution) for solution in self.pop])

    def computeValidArray(self):
        """
        Creates validity array for current population

        Returns
        -------
            validArr : np.array
        Contains validity of each solution, 0 or 1
        """

        return np.array([self.computeValid(solution) for solution in self.pop])

    def rouletteSelect(self, standardisedFitArr):
        """
        Selects an index based on probabilities of standardised fitness
        array using roulette selection

        Parameters
        ----------
        standardisedFitArr : np.array
            Contains standardised fitness values of all members of current
            population, such that all values sum to 1

        Returns
        -------
        idx : int
            Index of selected fitness value
        """

        # generate random value [0, 1)
        rand = self.rng.uniform()
        cumulative = 0
        # iteratively add each standardised fitness value until cumulative
        # value exceeds the random number and return selected index
        for idx, standardisedFit in enumerate(standardisedFitArr):
            cumulative += standardisedFit
            if rand < cumulative:
                break
        return idx

    def randomBenchmark(self, generations):
        """
        Repeatedly generates random solutions to benchmark GA performance.
        Creates (generations + 1) * population size number of solutions and
        records all their fitnesses. This can be used to gauge success of GA
        by answering the question: 'Can the GA find a better solution than
        that which can be found by random generation for equal number of
        fitness function evaluations?'

        Parameters
        ----------
        generations : int
            Number of populations to be generated
        """

        for gen in range(generations+1):
            self.pop = self.initPop()
            self.fitArr = self.computeFitnessArray()
            self.validArr = self.computeValidArray()
            self.recordHistory()
        return

    def initPop(self):
        """Problem specific, to be overwritten in child class"""
        pass

    def computeFitness(self):
        """Problem specific, to be overwritten in child class"""
        pass

    def computeValid(self):
        """Problem specific, to be overwritten in child class"""
        pass

    def performCrossover(self):
        """Problem specific, to be overwritten in child class"""
        pass

    def performMutation(self):
        """Problem specific, to be overwritten in child class"""
        pass
