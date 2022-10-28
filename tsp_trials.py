import numpy as np
rng = np.random.default_rng()

from geneticAlgorithm import *
from travellingSalespersonProblem import *

if __name__ == '__main__':

    plot = True
    run = True
    plotPath = True

    ### create problem graph ###
    nodes = 22

    # this graph cannot be plotted but can be directed
    # maxCost = 10
    # directed = True
    # graph = makeGraph(
    #     nodes,
    #     maxCost,
    #     directed,
    #     )
    # this graph can be plotted - costs are directly equivalent to distances
    bounds = np.array([10, 10])
    graph, locArr = makeRealGraph(numLocations=nodes, bounds=bounds)

    ### define trial ###
    numTrials = 2

    initialPopulation = [60]
    selectionMechanism = ['Roulette']
    crossoverPoints = ['random']
    elitism = [True]
    mutation = [True]
    mutationRate = [0.15, 0.25]
    permitInvalidSolutions = [False]

    trialVariable = mutationRate
    variableName = 'mutation rate'

    # create trial tsps
    trialsList = []
    for trial in range(numTrials):
        trialsList.append(TravellingSalespersonGA(
            initialPopulation[trial if len(initialPopulation) > 1 else 0],
            selectionMechanism[trial if len(selectionMechanism) > 1 else 0],
            crossoverPoints[trial if len(crossoverPoints) > 1 else 0],
            elitism[trial if len(elitism) > 1 else 0],
            mutation[trial if len(mutation) > 1 else 0],
            mutationRate[trial if len(mutationRate) > 1 else 0],
            permitInvalidSolutions[trial if len(permitInvalidSolutions) > 1 else 0],
            graph,
            ))

    if run:
        # perform evolution
        generations = 600
        earlyStopGenerations = 400
        for trial in range(numTrials):
            trialsList[trial].evolve(
            generations,
            earlyStopGenerations,
            )

        # compute problem complexity
        factorial = 1
        current = 1
        while current <= nodes:
            factorial = current * factorial
            current += 1

        # generate random benchmark solutions
        rand = TravellingSalespersonGA(
            max(initialPopulation),
            selectionMechanism[0],
            crossoverPoints[0],
            elitism[0],
            mutation[0],
            mutationRate[0],
            permitInvalidSolutions[0],
            graph,
            )
        rand.randomBenchmark(generations)

    if plot:
        maxFitnessesArr = np.array([np.amax(trial.fitArr[trial.validArr]) for trial in trialsList])
        fig, axs = plt.subplots(numTrials, 2, figsize=(10,2*numTrials))
        for trial in range(numTrials):
            # get best solution of final generation
            best = np.array(trialsList[trial].pop)[trialsList[trial].validArr][np.argmax(trialsList[trial].fitArr[trialsList[trial].validArr])]


            # plot evolution of best solution's fitness
            axs[trial, 0].plot(
                range(trialsList[trial].history.shape[0]),
                trialsList[trial].history[:,0],
                label='GA solution max',
                )
            # plot evolution of mean solution fitness
            axs[trial, 0].plot(
                range(trialsList[trial].history.shape[0]),
                trialsList[trial].history[:,1],
                label='GA solution mean')
            # plot maximum fitness found by random sampling
            axs[trial, 0].hlines(
                rand.fitArrHistory.max(),
                0,
                trialsList[trial].history.shape[0],
                linestyles='dotted',
                colors='k',
                label='Random max'
                )
            # plot mean fitness found by random sampling
            axs[trial, 0].hlines(
                rand.fitArrHistory.mean(),
                0,
                trialsList[trial].history.shape[0],
                linestyles='dashed',
                colors='k',
                label='Random mean'
                )
            # label plots with trial variable
            axs[trial, 0].set_title(f'{variableName}: {trialVariable[trial]}', loc='center')
            # share y axis for ease of fitness comparison
            if trial > 0:
                axs[trial,0].sharey(axs[0,0])
            if plotPath:
                # create array indicating path of best solution
                pathArray = np.zeros((nodes+1, bounds.shape[0]))
                for idx, loc in enumerate(best):
                    pathArray[idx] = locArr[loc]
            # plot locations
                axs[trial,1].scatter(
                    x=locArr[:,0],
                    y=locArr[:,1]
                )
                # plot path between locations
                axs[trial,1].plot(
                    pathArray[:,0],
                    pathArray[:,1],
                )

        fig.text(0.04, 0.1, f'With {nodes} locations\n\
there are {(factorial/nodes):.1e}\n\
possible unique paths.\n\n\
Max possible solutions\n\
evaluated: {(max(initialPopulation)*generations):.1e}\n\n\
Max fitness of \
{(np.amax(maxFitnessesArr)):.1e}\n\
achieved using\n\
{variableName}: {trialVariable[np.argmax(maxFitnessesArr)]}')

        axs[0,0].legend(loc=(-0.8, 0.3))
        plt.tight_layout()
        plt.show()
