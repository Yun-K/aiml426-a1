# %% [markdown]
# # Part 3 NSGA II
# 
# > Code is inspired from:
# > 
# > https://medium.com/@rossleecooloh/optimization-algorithm-nsga-ii-and-python-package-deap-fca0be6b2ffc
# >
# > https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
# >
# >  https://github.com/DEAP/deap/blob/master/deap/tools/emo.py

# %%
from matplotlib import pyplot
import pandas as pd
import numpy as np
from copy import deepcopy

# Import deque for the stack structure, copy for deep copy nodes
from collections import deque
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

# %%

# define some constants for the genetic algorithm

CONSTANTS_DICT = {
    "POPULATION_SIZE": 80, # number of individuals in each population
    "MAX_GENERATIONS": 66, # number of generations to run the algorithm
    "CROSSOVER_RATE": 0.9, # crossover rate should be very high, based on slides
    "MUTATION_RATE": 0.2, # mutation rate
    "CLASSIFIER":KNeighborsClassifier() , # classifier to use
    # "BOUND_LOW": 0.0, # lower bound for the features
    # "BOUND_UP": 1.0, # upper bound for the features
    # "ETA": 20.0, # crowding degree for mutation  and crossover
}


# %% [markdown]
# # dataset load

# %%
from queue import Empty


class DatasetPart3:
    def __init__(self, df) :
        self.df=df
        # self.df.columns = self.df.columns.str.strip()
        self.X = self.df.iloc[:,:-1]
        self.y = self.df.iloc[:,-1]
        # self.M = self.df.shape[0]  # number of rows
    
    # @classmethod
    # def constructFromFile(cls, filePath):
    #     """Depends on different ds"""
    #     pass

    def getDfWithSelectedFeatures(self, selectedFeatures):
        """No need to avoid FS bias, just based on df"""
        
        assert len(selectedFeatures) == self.X.shape[1]
        
        # according to the selected features, get the df with selected features
        colone_X = deepcopy(self.X)
        count, index_to_drop = 0, []
        for i in range(len(selectedFeatures)):
            isSelected = selectedFeatures[i] > 0.5 
            if  isSelected:
                index_to_drop.append(i)
                count += 1
        colone_X.drop(colone_X.columns[index_to_drop], axis=1, inplace=True)
        # count = len(selectedFeatures) - sum(selectedFeatures)
        # print(count )
        assert count ==  sum(selectedFeatures), f" {count} count != { sum(selectedFeatures)}"
        # assert count != 0, f"count is {count}"
        # assert count is 
        # concat the X and y
        returnedDf = colone_X.join(self.y)
        return returnedDf, count
            
        
        
        # NOTE: the same tihng as the above, can do either way 
        
        # returnedDf = pd.DataFrame()
        # selectedCount = 0
        # for i in range(len(selectedFeatures)):
        #     isSelected = True if selectedFeatures[i] > 0.5 else False
        #     if isSelected:
        #         selectedCount += 1
        #         # concat this feature to the returned dataframe
        #         returnedDf = pd.concat([returnedDf,self.df.iloc[:,i]],axis=1)
        # # concat the class column
        # returnedDf = pd.concat([returnedDf, self.y],axis=1)
        # assert returnedDf.empty == False

        # return returnedDf, selectedCount
    
    @staticmethod
    def run_model(df:pd.DataFrame, classifier=CONSTANTS_DICT["CLASSIFIER"]):

        assert df.empty == False
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=666)
        # classifier.fit(X_train, y_train)
        # return classifier.score(X_test, y_test)
    
        # cares ONLY about the training accuracy, so no need to split the data
        classifier.fit(X, y)
        return classifier.score(X, y)

        # # cv is super slow, so I use the train_test_split

        # # # evaluate the model
        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=666)
        # # return the error
        # n_scores = cross_val_score(classifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        # return np.mean(n_scores)
        
        

class Vehicle(DatasetPart3):
    def __init__(self, df):
        super().__init__(df)
    
    @classmethod
    def constructFromFile(cls, filePath):
        df = pd.read_csv(filePath, header=None, delim_whitespace=True)
        df.columns = [f"f_{i}" for i in range(len(df.columns))]
        df.rename(columns = {f'f_{len(df.columns)-1}':'class'}, inplace = True)
        return cls(df)
    
class MuskClean(DatasetPart3):
    def __init__(self, df):
        super().__init__(df)

    @classmethod
    def constructFromFile(cls, filePath):
        df = pd.read_csv(filePath, header=None)
        # ignore the first 2 columns since they are NOT numerical, so it would be betteer to ignore them 
        df.drop([0,1], axis=1, inplace=True)
        df.columns = [f"f_{i}" for i in range(len(df.columns))]
        df.rename(columns = {f'f_{len(df.columns)-1}':'class'}, inplace = True)
        return cls(df)
    


# %%
ds_vehicle = Vehicle.constructFromFile("./vehicle/vehicle.dat")
print(ds_vehicle.X.shape[1])
# ds_vehicle.df.info()
ds_vehicle.df

# %%
# ds_mushclean = MuskClean.constructFromFile("./musk/clean1.data")
# ds_mushclean.df
# # # len(ds_mushclean.x.columns)

# %% [markdown]
# # set up creator

# %%
# 2 minimum objectives, so -1,-1
creator.create("MultiObjMin", base.Fitness, weights=(-1.0, -1.0)) 
# Individual should be a list of binary values, i.e. a list of 0s and 1s
creator.create("Individual", list, fitness=creator.MultiObjMin)

# %% [markdown]
# define wrapper based fitness evaluate function

# %%
def wrapperFitnessEvaluation(ds:DatasetPart3, individual:creator.Individual, 
                             classifier=CONSTANTS_DICT["CLASSIFIER"]): #KNN by default
    df_selected,selected_count = ds.getDfWithSelectedFeatures(individual)
    # df_selected = getTransformedDf(df_selected)
    
    if selected_count == 0:
        return 99999, 99999 # return a very large number to penalty no feature selected
    
    
    
    acc_score = DatasetPart3.run_model(df_selected, classifier)
    obj1 = 1.0-acc_score # classification error
    obj2 = selected_count / len(individual) #ratio of selected features
    assert 0<=obj1<=1 and 0<=obj2<=1
    return obj1, obj2

# %% [markdown]
# tool box
# 
# > https://www.researchgate.net/publication/235707001_DEAP_Evolutionary_algorithms_made_easy

# %%
# from deap import dtm
# toolbox is a class contains the operators that we will use in our genetic programming algorithm
# it can be also be used as the container of methods which enables us to add new methods to the toolbox 
def setup_toolbox(ds:DatasetPart3, randSeed:int) -> base.Toolbox:
    toolbox = base.Toolbox()
    # toolbox.register("map",dtm.map)
    
    # for population size, we use the random.randint function to generate a random integer in the range [min, max]
    random.seed(randSeed)
    # register a method to generate random boolean values
    toolbox.register("attr_bool", random.randint, 0, 1)
    # register a method to generate random individuals
    toolbox.register("IndividualCreator", 
                     tools.initRepeat, 
                     creator.Individual, 
                     toolbox.attr_bool, 
                     n=len(ds.X.columns) # feature number, exclude the class column
                    )
    
    # N is not specificied, so need to specify number of individuals to generate within each population when we call it later
    toolbox.register("PopulationCreator", tools.initRepeat, list, toolbox.IndividualCreator) 

    # toolbox.register("select", tools.emo.selTournamentDCD)
    toolbox.register("select", tools.selNSGA2)
    # toolbox.register('selectGen1', tools.selTournament, tournsize=2)
    
    
    
    # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=CONSTANTS_DICT["BOUND_LOW"], up=CONSTANTS_DICT["BOUND_UP"], eta=CONSTANTS_DICT["ETA"])
    # toolbox.register("mutate", tools.mutPolynomialBounded, low=CONSTANTS_DICT["BOUND_LOW"], up=CONSTANTS_DICT["BOUND_UP"], eta=CONSTANTS_DICT["ETA"], indpb=1.0/len(ds.x.columns))
    

    
    toolbox.register("mate", tools.cxTwoPoint)
    # indpb refer to the probability of mutate happening on each gene, it is NOT the same as mutation rate
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(ds.X.columns)) 
    
    toolbox.register("evaluate", wrapperFitnessEvaluation, ds) # need to pass individual:list
    return toolbox

# %% [markdown]
# # run NSGA once  asas
# 
# > https://github.dev/DEAP/deap/blob/master/deap/tools/emo.py
# > https://github.dev/DEAP/deap/blob/master/examples/ga/nsga2.py

# %%
import copy
from select import select
import time

def run_NSGAII(ds:DatasetPart3, randSeed:int, 
                ngen:int=CONSTANTS_DICT["MAX_GENERATIONS"], 
                popSize:int=CONSTANTS_DICT["POPULATION_SIZE"]):
    # stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("mean", np.mean, axis = 0)
    stats.register("std", np.std, axis=0)
    # for record keeping
    logbook = tools.Logbook()    
    logbook.header = "gen", "mean", "std", "min", "max"
    
    # create toolbox
    random.seed(randSeed)
    toolbox = setup_toolbox(ds, randSeed)
    
    # calculate objectives
    def evaluate_update_fitness_values(p) :
        """Update the fitness values of each individual for the given the population"""
        fitnesses = list(map(toolbox.evaluate, p))
        for ind, fit in zip(p, fitnesses):
            ind.fitness.values = fit

             
     # initial population 
    pop = toolbox.PopulationCreator(n=popSize)
    evaluate_update_fitness_values(pop)
    
    
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    record = stats.compile(pop)
    logbook.record(gen=0, **record)
    print(logbook.stream)
    
    # # fast non dominated sort
    # fronts = tools.emo.sortNondominated(pop,k=popSize)
    # for idx,front in enumerate(fronts):
    #     #print(idx,front)
    #     for ind in front:
    #         ind.fitness.values = (idx+1),# change fitness to the order of pareto front
    
    # #  generate offspring
    # offspring = toolbox.selectGen1(pop, len(pop))
    # # apply mate and mutate only once
    # offspring = algorithms.varAnd(offspring,toolbox,
    #                               CONSTANTS_DICT["CROSSOVER_RATE"],
    #                               CONSTANTS_DICT["MUTATION_RATE"]) 

    # Begin the generational process
    for gen_counter in range(1,ngen):
        
        # Vary the pop
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # apply mate and mutate , clone the individual, 
        # so returned offspring is independent of input parents
        offspring = algorithms.varAnd(offspring, toolbox, 
                                      cxpb=CONSTANTS_DICT["CROSSOVER_RATE"], 
                                      mutpb=CONSTANTS_DICT["MUTATION_RATE"])
           
        # Evaluate all  offsprings individuals 
        combined_pop = offspring + pop
        evaluate_update_fitness_values(combined_pop)
      
        # elitism strategy
        # Select the next generation pop
        pop = toolbox.select(combined_pop, popSize)
        # stats
        record = stats.compile(pop)
        logbook.record(gen=gen_counter,  **record)
        print(logbook.stream)
        
    print("Final pop hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))
    return pop, logbook, hypervolume(pop, [11.0, 11.0]) # pop is a set of non-dominated individuals solutions

    # # create the initial population
    # pop = toolbox.PopulationCreator(n=popSize)
    # evaluate_update_fitness_values(pop)
    # for g in range(ngen):
    #     offspring = algorithms.varAnd(pop, toolbox, 
    #                                   cxpb=CONSTANTS_DICT["CROSSOVER_RATE"], 
    #                                   mutpb=CONSTANTS_DICT["MUTATION_RATE"])
    #     evaluate_update_fitness_values(offspring)
    #     pop = toolbox.select(pop + offspring, k=len(pop))
    #     record = stats.compile(pop)
    #     logbook.record(gen=g,  **record)
    #     print(logbook.stream)
    

   

# %% [markdown]
# Another NSGA 2 implementation from 
# 
# > https://blog.csdn.net/weixin_46649052/article/details/110230757

# %%
import copy
from select import select
import time


def run_NSGAII_2(ds:DatasetPart3, randSeed:int, 
                ngen:int=CONSTANTS_DICT["MAX_GENERATIONS"], 
                popSize:int=CONSTANTS_DICT["POPULATION_SIZE"]):
    # stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("mean", np.mean, axis = 0)
    stats.register("std", np.std, axis=0)
    # for record keeping
    logbook = tools.Logbook()    
    logbook.header = "gen", "mean", "std", "min", "max"
    
    # create toolbox
    random.seed(randSeed)
    toolbox = setup_toolbox(ds, randSeed)
    
    # calculate objectives
    def evaluate_update_fitness_values(p) :
        """Update the fitness values of each individual for the given the population"""
        fitnesses = list(map(toolbox.evaluate, p))
        for ind, fit in zip(p, fitnesses):
            ind.fitness.values = fit
             
     # initial population 
    pop = toolbox.PopulationCreator(n=popSize)
    evaluate_update_fitness_values(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, **record)
    print(logbook.stream)
    
    # fast non dominated sort
    fronts = tools.emo.sortNondominated(pop,k=popSize)
    for idx,front in enumerate(fronts):
        #print(idx,front)
        for ind in front:
            ind.fitness.values = (idx+1),# change fitness to the order of pareto front
    
    #  generate offspring
    offspring = tools.selTournament(pop,len(pop), tournsize=2 )
    # offspring = toolbox.selectGen1(pop, len(pop))
    # apply mate and mutate only once
    offspring = algorithms.varAnd(offspring,toolbox,
                                  CONSTANTS_DICT["CROSSOVER_RATE"],
                                  CONSTANTS_DICT["MUTATION_RATE"]) 

    # Begin the generational process
    for gen_counter in range(1,ngen):
        # Evaluate all  offsprings individuals 
        combined_pop = offspring + pop
        evaluate_update_fitness_values(combined_pop)
        
        # fast non dominated sort
        fronts =  tools.emo.sortNondominated(combined_pop,k=popSize,first_front_only=False)
        
        # crowding distance
        for front in fronts:
            tools.emo.assignCrowdingDist(front)
            
        # environmental selection -- elitism strategy
        pop = []
        for front in fronts:
            pop +=front
            
        # use the select function that based on crowding distance to implement elitism strategy
        pop = tools.selNSGA2(toolbox.clone(pop), popSize)
        
        # generate offspring
        offspring = tools.selTournamentDCD(pop, popSize)
        offspring = toolbox.clone(offspring)
        offspring = algorithms.varAnd(offspring, toolbox,
                                      cxpb=CONSTANTS_DICT["CROSSOVER_RATE"],
                                      mutpb=CONSTANTS_DICT["MUTATION_RATE"])
        
        # stats
        record = stats.compile(pop)
        logbook.record(gen = gen_counter,**record)
        print(logbook.stream)
        
        
    print("Final pop hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))
    return pop, logbook, hypervolume(pop, [11.0, 11.0]) # pop is a set of non-dominated individuals solutions



   

# %% [markdown]
# # run 3 times

# %%
import matplotlib.pyplot as plt
def run_3_times_with_different_seed(ds:DatasetPart3,
                                     title:str, 
                                     max_gen=CONSTANTS_DICT["MAX_GENERATIONS"],
                                     classifier = CONSTANTS_DICT["CLASSIFIER"],
                                     randSeed = [i+1 for i in range(3)],
                                     run_times=3):
    # run 3 times with different seed
    population_list = []
    logbook_list = []
    hypervolume_list = []
    
    color_list = ['r.','g.','b.','c.','m.','y.','k.']
    front_list = []
    for i in range(run_times):
        print('-'*80)
        print('-'*80)
        print(title,"\nRunning GA with seed: ", randSeed[i])
        population, logbook, hypervolume = run_NSGAII(ds, randSeed=randSeed[i], ngen=max_gen, popSize=CONSTANTS_DICT["POPULATION_SIZE"])
        population_list.append(population)
        logbook_list.append(logbook)
        hypervolume_list.append(hypervolume)    
        print('-'*80)
        print('-'*80)
        
        # # plot the result
        front = tools.emo.sortNondominated(population,len(population),first_front_only=True)[0]
        err_rate = [ind.fitness.values[0] for ind in front]
        ratio_selcted = [ind.fitness.values[1] for ind in front]
        # for ind in front:
        plt.plot(err_rate, ratio_selcted, color_list[i] , ms=2,
                 label = f"seed {randSeed[i]},hypervolume: {hypervolume}")
        plt.ylabel("ratio of selected features")
        plt.xlabel("classification error rate")
        plt.title(f"dataset: {title}, seed: {randSeed[i]}\nhypervolume: {hypervolume}\n Objective space")
        # plt.tight_layout()
    # plt.legend(bbox_to_anchor =(1.3,-0.1), loc='lower center')
        plt.show()
        print(len(front))
        
        # append front to front_list, remove duplicates
        front = [ind.fitness.values for ind in front]
        front = list(set(front))
        front_list.append(front)
        print(len(front))
        
      
    # compare error rates of the obtained solutions with that of using the entire feature set.
    subset_mean_err_rate = [logbook.select("min")[-1][0] for logbook in logbook_list]
    entire_mean_err_rate = 1-DatasetPart3.run_model(ds.df)
    
    print(f"{title}:\n error rate of using the entire feature set: {entire_mean_err_rate}")
        
    print(f"hypervolume of the obtained solution for each run : {hypervolume_list}")

    return population_list, logbook_list, hypervolume_list, front_list

# %%
def print_pareto_optimal_solutions_details(front_list:list):
  print(f"Pareto-optimal solutions : ")
  # remove duplicates
  # front_list = [list(t) for t in set(tuple(element) for element in front_list)]
  
  for i in range(len(front_list)) :
      front = front_list[i]
      
      print("-"*60)
      print(f"|Pareto-optimal front individuals for run with seed {i+1}:| error rate| ratio of selected features |")
      for j in range(len(front)):
          ind = front[j]
          print(f"|Individual: {j} | {ind[0]} | {ind[1]}|")
          # print(f"|Individual: {j} | {ind.fitness.values[0]} | {ind.fitness.values[1]}|")
                                  
  print("-"*80)
  

# %%
ds_vehicle = Vehicle.constructFromFile("./vehicle/vehicle.dat")
pop_list_vehicle,logbook_list_vehicle, hypervolume_list_vehicle, front_list_vehicle = run_3_times_with_different_seed(ds_vehicle, "vehicle",
                                max_gen=66,
                                run_times=3)

print_pareto_optimal_solutions_details(front_list_vehicle)

# %%

ds_mushclean = MuskClean.constructFromFile("./musk/clean1.data")
pop_list_musk,logbook_list_musk, hypervolume_list_musk, front_list_musk= run_3_times_with_different_seed(ds_mushclean, 
                                                                          "muskclean",
                                                                          max_gen=66,
                                                                          run_times=3)
print_pareto_optimal_solutions_details(front_list_musk)


