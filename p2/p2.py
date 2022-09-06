# %% [markdown]
# For reading the dataset 

# %%
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import re
import pandas as pd
import numpy as np
from copy import deepcopy
from distutils.command.build_scripts import first_line_re
from tkinter.tix import COLUMN
# Import deque for the stack structure, copy for deep copy nodes
from collections import deque
from sklearn.metrics import accuracy_score
import sklearn 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          ExtraTreeClassifier)
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
# Encoding categorical features with preserving the missing values in incomplete features
from sklearn.preprocessing import (KBinsDiscretizer, LabelEncoder,
                                   OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)

# define some constants for the genetic algorithm
CONSTANTS_DICT = {
    "POPULATION_SIZE": 100, # number of individuals in each population
    "MAX_GENERATIONS": 250, # number of generations to run the algorithm
    "CROSSOVER_RATE": 1.0, # crossover rate should always be 100%, based on slides
    "MUTATION_RATE": 0.2, # mutation rate
    "ELITIST_PERCENTAGE": 0.05, # percentage of the best individuals to keep in the next generation
    "CLASSIFIER": DecisionTreeClassifier(criterion='entropy'), # classifier to use
}


# %%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
class DatasetPart2:
    
    def summarize_ds(self):
        print(self.df.shape)
        # summarize each variable
        print(self.df.describe())
        # histograms of the variables
        self.df.hist()
        pyplot.show()

    @staticmethod
    def run_model(df:pd.DataFrame, classifier):
        x = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        
        # evaluate the model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(classifier, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        # report model performance
        
        
        # classifier.fit(x,y)
        # acc_score = accuracy_score(y, classifier.predict(x))
        return np.mean(n_scores)

    
 

    def __init__(self,df:pd.DataFrame):
        self.df=df
        self.x = self.df.iloc[:,:-1]
        self.y = self.df.iloc[:,-1]
        self.M = self.df.shape[0]  # number of rows
        # for avoiding FS bias
        self.df_fs = df.iloc[:int(0.7*self.M),:]
    
    def getDfWithSelectedFeatures(self, selectedFeatures:list):
        """For avoiding FS bias, df_fs is used instead of df to obtain the selected features"""
        returnedDf = pd.DataFrame()
        for i in range(len(selectedFeatures)):
            isSelected = True if selectedFeatures[i] == 1 else False
            if isSelected:
                # concat this feature to the returned dataframe
                returnedDf = pd.concat([returnedDf,self.df_fs.iloc[:,i]],axis=1)
        # concat the class column
        returnedDf = pd.concat([returnedDf,self.df_fs.iloc[:,-1]],axis=1)
        return returnedDf
    
    @classmethod
    def constructFromFile(cls,filePath):
        df = pd.read_csv(filePath,header=None)
        df.columns = [f"f_{i}" for i in range(len(df.columns))]
        df.rename(columns = {f'f_{len(df.columns)-1}':'class'}, inplace = True)
        return cls(df) 
    
    @staticmethod
    def getTransformedDf(df2Transform:pd.DataFrame):
        """transform the continous features to discontinous. In other words, due to all features are continous, this functions are used to discretise all continous features.

        KBins is used to discretise the continous features. The number of bins is set to 10. The strategy is set to uniform.
        
        Tutorial: https://machinelearningmastery.com/discretization-transforms-for-machine-learning/
        
        Args:
            df2Transform (pd.DataFrame): df to transform, all features should be continous
            
        """ 
        tempDf = deepcopy(df2Transform)
        tempDf_x = tempDf.iloc[:,:-1]
        tempDf_y = tempDf.iloc[:,-1]
        # only transform the continous features, ignore Y
        kbins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        tempDf_x = kbins.fit_transform(tempDf_x)
        tempDf = pd.concat([pd.DataFrame(tempDf_x),tempDf_y],axis=1)
        tempDf.columns = [f"f_{i}" for i in range(len(tempDf.columns))]
        tempDf.rename(columns = {f'f_{len(tempDf.columns)-1}':'class'}, inplace = True)
        return tempDf
        
class Sonar(DatasetPart2):
    def __init__(self,df):
        super().__init__(df)
    
class Wbcd(DatasetPart2):
    def __init__(self,df):
        super().__init__(df)

ds_sonar  = Sonar.constructFromFile("./sonar/sonar.data")
ds_wbcd = Wbcd.constructFromFile("./wbcd/wbcd.data")

# # ds_sonar.getDfWithSelectedFeatures([0,1,1,1,1,0])
# ds_sonar.getTransformedDf(ds_sonar.df)
# ds_sonar.summarize_ds()
# ds_sonar.df


ds_sonar.M

# %%
testDf = ds_sonar.getTransformedDf(ds_sonar.df)
testDf


# %%
# # ds_sonar.df
# testDf_x = testDf.iloc[:,:-1]
# unique, count = np.unique(testDf_x, return_counts=True, axis=0)
# a = count / len(testDf_x)
# len(a)
# # np.sum(a)

# %%
# # ds_sonar.df
# testDf_x = testDf.iloc[:,:-1]
# unique, count = np.unique(testDf_x.iloc[:,0], return_counts=True, axis=0)
# a = count / len(testDf_x.iloc[:,0])
# # np.sum(a)
# a

# %%
# # ds_sonar.df
# testDf_x = testDf.iloc[:,:-1]
# testDf_y = testDf.iloc[:,-1]
# unique, count = np.unique(testDf_x.iloc[:,0], return_counts=True, axis=0)
# prob = count / len(testDf_x.iloc[:,0])
# # np.sum(a)
# # len(unique)
# prob
# {unique:prob for unique,prob in zip(unique,prob)}

# # 1st row of x
# # testDf_x.iloc[0,:]
# Y = testDf_y
# y_unique, y_count = np.unique(Y, return_counts=True, axis=0)
# p_Y = y_count/len(Y)
# p_Y_dict = {y_unique:p_Y for y_unique,p_Y in zip(y_unique,p_Y)}
# p_Y_dict[Y[207]]
# # Y[207]

# %% [markdown]
# > Use **Creator** to define the type of individuals and fitness classes.

# %%
from asyncio import constants
from json import tool
from deap import creator, base, gp, tools, algorithms # core functionality of DEAP
import array
import random
import json
import math # for checking the fitness of an individual, i.e. math.isinf(weight)
import matplotlib.pyplot as plt
# Python internal operators for object comparisons, 
# logical comparisons, arithmetic operations, and sequence operations
import operator 

# creator is  usually used to define the type of the individual and fitness classes

# goal:to maximize the value and do not exceed the capacity of the knapsack
# define strategies with different priorities for optimizing multiple goals by using FitnessCompound
# 1 for maximize value, -1 for minimize weight, 
# creator.create("FitnessCompound", base.Fitness, weights=(1.0,-1.0)) 
 
# according to slide, fitness value has been reduced to 1 dimension, so just use FitnessMax
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Individual should be a list of binary values, i.e. a list of 0s and 1s
creator.create("Individual", list, fitness=creator.FitnessMax)


# %% [markdown]
# > Define the evaluate function for **FilterGA** and **WrapperGA**.
# > 
# > Inspired by the slide, and https://datascience.stackexchange.com/questions/58565/conditional-entropy-calculation-in-python-hyx
# >
# > code is from https://datascience.stackexchange.com/questions/58565/conditional-entropy-calculation-in-python-hyx

# %%
##Entropy
from calendar import c


def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en
    # my implementation, it is the same, 
    # entropy_y = 0
    # for i in range(len(Y.unique())):
    #     p_y = len(Y[Y==Y.unique()[i]])/len(Y)
    #     entropy_y += -p_y*math.log2(p_y)
    # return entropy_y
    
# print(entropy(ds_sonar.y))


# %%
# import numpy as np
# %load_ext cython
def conditional_entropy_python(X, Y):
    """ 
    Calculate conditional entropy of all columns of X against Y (i.e. \sum_i=1^{N} H(X_i | Y)).
    https://gist.github.com/kudkudak/dabbed1af234c8e3868e
    """
    # Calculate distribution of y    
    Y_dist = np.zeros(shape=(int(Y.max()) + 1, ), dtype=np.float32)
    for y in range(Y.max() + 1):
        Y_dist[y] = (float(len(np.where(Y==y)[0]))/len(Y))
        
    Y_max = Y.max()
    X_max = X.max()
    
    ce_sum = 0.
    for i in range(X.shape[1]):
        ce_sum_partial = 0.
        
        # Count 
        counts = np.zeros(shape=(X_max + 1, Y_max + 1), dtype=np.float32)
        for row, x in enumerate(X[:, i]):
            counts[x, Y[row]] += 1
        
        # For each value of y add conditional probability
        for y in range(Y.max() + 1):
            count_sum = float(counts[:, y].sum())
            probs = counts[:, y] / count_sum
            entropy = -probs * np.log2(probs)
            ce_sum_partial += (entropy * Y_dist[y]).sum()

        ce_sum += ce_sum_partial
        
    return ce_sum

# %%

#Joint Entropy
def jEntropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y,X]
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)

    slide:
    H(Y|X=(X1,..,Xm)) = sum( p(x1,...,xm) H(Y|X1=x1,X2=x2,...,Xm=xm)) )
                    = - sum(sum(p(x1,..,xm) * p(y|x1,..,xm) * log2(p(y|x1,..,xm))))
    
    
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    # return conditional_entropy_python(X, Y)
    # return jEntropy(Y, X) - entropy(X)
    
    # slide:
    # H(Y|X=(X1,..,Xm)) = sum( p(x1,...,xm) H(Y|X1=x1,X2=x2,...,Xm=xm)) )
    #                 = - sum(sum(p(x1,..,xm) * p(y|x1,..,xm) * log2(p(y|x1,..,xm))))

    # assign the probability of each value of X to a variable
    p_X_list = []
    for i in range(len(X.columns)):
        unique, count = np.unique(X.iloc[:,i], return_counts=True, axis=0)
        prob_x_i = count / len(X.iloc[:,i])
        p_X_list.append({unique:prob_x_i for unique,prob_x_i in zip(unique,prob_x_i)})
        
    # p(Y)
    y_unique, y_count = np.unique(Y, return_counts=True, axis=0)
    p_Y = y_count/len(Y)
    p_Y_dict = {y_unique:p_Y for y_unique,p_Y in zip(y_unique,p_Y)}
    
    # iterare through all rows
    cEn = 0
    for i in range(len(X)):
        # cols are independent, so we can multiply the probabilities
        # p(x1,..,xm) = p(x1)*p(x2)*...*p(xm)
        row_i = X.iloc[i,:]
        p_Xi = 1
        for j in range(len(row_i)):
            p_Xi *= p_X_list[j][row_i[j]]
        
        # p(y|x1,..,xm) = p(x1,..,xm | y) p(y)  /p(x1,..,xm)
        p_YX = 0
        for i in range(len(y_unique)):
            p_YX += p_Y_dict[y_unique[i]] * p_Xi
            # p_YX += p_Y_dict[Y[i]] * p_Xi / p_Xi
        # p(x1,..,xm) * p(y|x1,..,xm) * log2(p(y|x1,..,xm))
        cEni = - p_Xi *  p_YX * np.log2(p_YX)
        cEn += cEni
    # print(cEn)
    return cEn 
        
    
    
    
    
    

    
    # # - sum(sum(p(x1,..,xm) * p(y|x1,..,xm) * log2(p(y|x1,..,xm))))
    # cEn = 0
    # for i in range(len(X.columns)):
    #     for j in range(len(X.iloc[:,i].unique())):
    #         # p(x1,..,xm)
    #         p_X = 1
    #         for k in range(len(p_X_list)):
    #             if k != i:
    #                 p_X *= p_X_list[k][j]
    #             else:
    #                 p_X *= p_X_list[k][j]/len(X.iloc[:,i].unique())
    #         # p(y|x1,..,xm)
    #         p_Y_X = np.zeros(shape=(len(Y.unique()),), dtype=np.float32)
    #         for k in range(len(Y.unique())):
    #             p_Y_X[k] = len(Y[(Y==Y.unique()[k]) & (X.iloc[:,i]==X.iloc[:,i].unique()[j])])/len(Y)
    #         # p(y|x1,..,xm) * log2(p(y|x1,..,xm))
    #         p_Y_X_log = np.zeros(shape=(len(Y.unique()),), dtype=np.float32)
    #         for k in range(len(Y.unique())):
    #             if p_Y_X[k] != 0:
    #                 p_Y_X_log[k] = p_Y_X[k] * math.log2(p_Y_X[k])
    #         # - sum(sum(p(x1,..,xm) * p(y|x1,..,xm) * log2(p(y|x1,..,xm))))
    #         cEn += - p_X * np.sum(p_Y_X_log)
            
    # return cEn
    
    
    
    
    
    # for i in range(len(p_X)):
    #     for j in range(len(p_Y)):
    #         # Bayes rule and product rule
    #         # p(y|x1,..,xm) = p(y,x1,..,xm)/p(x1,..,xm) 
    #         p_yx = p_X[i]*p_Y[j] / p_X[i]
    #         cEn += -p_X[i]*p_yx*math.log2(p_yx)
            
            
            
    #         # if p_X[i]*p_Y[j] != 0:
    #         #     cEn += -p_X[i]*p_Y[j]*math.log2(p_X[i]*p_Y[j])
    # return cEn
    
    
    
    
    # for i in range(len(y_unique)):
    #     for j in range(len(p_X)):
    #         # Bayes rule and product rule
    #         # p(y|x1,..,xm) = p(y,x1,..,xm)/p(x1,..,xm) 
    #         p_YX = p_X[j] * p_Y[i] / p_X[j]
    #         cEn += p_X * p_YX * math.log2(p_YX/p_Y)
    # return cEn
        
    
    
    
    # cEntropy = 0
    # # for i in range(len(X.columns)):
    # #     for j in range(len(X[X.columns[i]].unique())):
    # #         p_x = len(X[X[X.columns[i]]==X[X.columns[i]].unique()[j]])/len(X)
    # #         p_yx = len(Y[X[X.columns[i]]==X[X.columns[i]].unique()[j]])/len(X[X[X.columns[i]]==X[X.columns[i]].unique()[j]])
    # #         cEntropy += -p_x*p_yx*math.log2(p_yx)
    
    # for i in range(len(X.columns)):
    #     x_i = X.iloc[:,i]
    #     entropy_x_i = entropy(x_i)
    #     jEntropy_y_xi = jEntropy(Y,x_i)
    #     cEntropy_i = jEntropy_y_xi - entropy_x_i
    #     cEntropy += cEntropy_i
    
    
    # return cEntropy
    


#Information Gain
def gain(Y, X):
    """
    Information Gain, I(Y;X) = H(Y) - H(Y|X)
    Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition
    """
    return entropy(Y) - cEntropy(Y,X)

# %%

def evaluateFilterGA(ds:DatasetPart2, individual:creator.Individual): 
    """Goodness of a individual(i.e. feature subset) independent of the classifier.
    Information Gain is used to evaluate the goodness of a feature subset.

    Args:
        ds (DatasetPart2): Given ds, for which the individual is evaluated
        individual (creator.Individual): _description_
    """ 
    # get the df with selected features, StandardScaler is already used to scale continous features into the discrete values
    df_selected = ds.getDfWithSelectedFeatures(individual)
    df_selected_transformed = DatasetPart2.getTransformedDf(df_selected)
    y = df_selected_transformed.iloc[:,-1]
    x = df_selected_transformed.iloc[:,:-1]
    
    info_gain = gain(y,x) # I(Y;X) = H(Y) - H(Y|X)
    # info_gain_ratio = info_gain/entropy(x) # I(Y;X)/H(X)
    
    return info_gain / len(x.columns), #info_gain_ratio,

def evaluateWrapperGA(ds:DatasetPart2, individual:creator.Individual, classifier = CONSTANTS_DICT["CLASSIFIER"]): 
    df_selected = ds.getDfWithSelectedFeatures(individual)
    df_selected_transformed = DatasetPart2.getTransformedDf(df_selected)
    acc_score = DatasetPart2.run_model(df_selected_transformed, classifier)
    return acc_score,    

# %% [markdown]
# Setup toolbox for registering the functions

# %%

# toolbox is a class contains the operators that we will use in our genetic programming algorithm
# it can be also be used as the container of methods which enables us to add new methods to the toolbox 
def setup_toolbox(ds:DatasetPart2, evaluateFunction,randSeed:int) -> base.Toolbox:
    toolbox = base.Toolbox()
    # for population size, we use the random.randint function to generate a random integer in the range [min, max]
    random.seed(randSeed)
    # register a method to generate random boolean values
    toolbox.register("attr_bool", random.randint, 0, 1)
    # register a method to generate random individuals
    toolbox.register("IndividualCreator", 
                     tools.initRepeat, 
                     creator.Individual, 
                     toolbox.attr_bool, 
                     n=len(ds.x.columns) # feature number, exclude the class column
                    )
    
    # N is not specificied, so need to specify number of individuals to generate within each population when we call it later
    toolbox.register("PopulationCreator", tools.initRepeat, list, toolbox.IndividualCreator) 
    
    toolbox.register("elitism", tools.selBest, k=int(CONSTANTS_DICT["ELITIST_PERCENTAGE"]*ds.M))
    toolbox.register("select", tools.selTournament, k=2, tournsize=3)
    
    toolbox.register("mate", tools.cxTwoPoint) # TODO: might need to change this to cxOnePoint
    # indpb refer to the probability of mutate happening on each gene, it is NOT the same as mutation rate
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ds.M) # TODO: might need to change this to mutUniformInt
    # local search operator
    # toolbox.register("local_search", algorithms)
    
    
    # register the evaluation function
    toolbox.register("evaluate", evaluateFunction, ds) # register a method to evaluate the fitness of an individual
    return toolbox

# %%

import copy
from select import select
import time

def run_GA_framework(ds:DatasetPart2,evaluateFunction, max_gen= CONSTANTS_DICT["MAX_GENERATIONS"], randSeed:int=1) -> creator.Individual:
    '''
    Run the genetic algorithm framework
    '''
    # for toolbox
    random.seed(randSeed)
    toolbox = setup_toolbox(ds,evaluateFunction, randSeed)
    # for record keeping
    logbook = tools.Logbook()    
    # assign the stats for recording the computational time
    # stats = tools.Statistics()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("mean", np.mean, axis = 0)
    stats.register("std", np.std, axis=0)
  
    # create the initial population
    population = toolbox.PopulationCreator(n=CONSTANTS_DICT["POPULATION_SIZE"])
    
    # # evaluate the fitness of the current population, and assign the fitness to each individual
    # evaluate_fitness_values(population)
    def evaluate_fitness_values(pop) :
        """Update the fitness values of each individual for the given the population"""
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
    
    
    best_feasible_individual = None
    # computation_time_list = [] 
    startTime = time.time()
    # start the evolution
    for gen_counter in range(max_gen):
        # assign the fitness to each individual in the current generation 
        evaluate_fitness_values(population)
        # for recording the time spent on each generation
        genStartTime = time.time()
        # # time_cost = computation_time_list[-1] - genStartTime if len(computation_time_list) > 0 \
                                                            #  else 0.0
        # computation_time_list.append(time_cost)
        # for visualizing whether the algorithm is still running
        best_feasible_individual = tools.selBest(population, k=5)[0]
        best_fintess_current_gen = best_feasible_individual.fitness.values[0]
        # record the statistics of the current generation
        # record = stats.compile(computation_time_list) 
        # logbook.record(gen=gen_counter,
        #                best_fintess_current_gen=best_fintess_current_gen, best_ind_chromosome=best_feasible_individual,computation_time_list=computation_time_list,
        #                **record)
        record = stats.compile(population)
        logbook.record(gen=gen_counter,
                       best_fintess_current_gen=best_fintess_current_gen, best_ind_chromosome=best_feasible_individual,
                       **record)
        
        
        # apply elitism to obtain the best individuals in the current generation
        offspring = toolbox.elitism(population)

        # repeat until the offspring has the same size as the population
        while len(offspring) < CONSTANTS_DICT["POPULATION_SIZE"]:
            # apply selection
            parent1,parent2 = toolbox.select(population)

            # apply crossover
            c1,c2 = toolbox.mate(copy.deepcopy(parent1),copy.deepcopy(parent2))
            
            # apply mutation to the children
            for child in [c1,c2]:
                if random.random() < CONSTANTS_DICT["MUTATION_RATE"]:
                    toolbox.mutate(child)
                    del child.fitness.values
                # append the children to the offspring
            # TODO: apply local search to the children
            # annoying, time consuming, not implemented yet although Yi's github got tutorial

            offspring.append(c1)
            offspring.append(c2)
        # replace the current population with the offspring new gwneration
        population[:] = offspring
        
    timeSpent = time.time() - startTime
    return best_feasible_individual, logbook, stats, timeSpent


# %%
def run_5_times_with_different_seeds(ds:DatasetPart2,
                                     title:str, 
                                     evaluateFunction,
                                     classifier = CONSTANTS_DICT["CLASSIFIER"],
                                     max_gen=CONSTANTS_DICT["MAX_GENERATIONS"],
                                     randSeed = [i+1 for i in range(5)],
                                     run_times=5):
    '''
    Run the GA framework 5 times with different seeds
    '''
    five_computional_time_list = []
    five_best_individual_list = []
    
    for i in range(run_times):
        best_feasible_individual,logbook,stats,timeSpent = run_GA_framework(ds,evaluateFunction, max_gen,randSeed[i])
        # assign best chromosome in order for applying the model: e.g. Navie Bayes 
        five_best_individual_list.append(best_feasible_individual)
        # assign the mean and std
        # five_computional_time_dict["mean"].append( logbook.select("mean")[-1])
        # five_computional_time_dict["std"].append( logbook.select("std")[-1])
        five_computional_time_list.append(timeSpent)

        
        
        print('-'*80)
        print('-'*80)
        print("Running GA with seed: ", randSeed[i])
        print('Best Individual fitness: ', best_feasible_individual.fitness.values[0])
        print("FOllowing are the statistics for each generation with the seed: ", randSeed[i])
        print('-'*80)
        logbook.header = "gen", "mean", "std", "best_fintess_current_gen","best_ind_chromosome"
        print(logbook)
        print('-'*80)
        print('-'*80)
    
    # transform the selected features by removing unused features
    # then apply the model to the selected features
    five_acc_score_list = []
    for i in range(len(five_best_individual_list)):
        df_selected = ds.getDfWithSelectedFeatures(five_best_individual_list[i])
        df_selected_transformed = DatasetPart2.getTransformedDf(df_selected)
        acc_score = DatasetPart2.run_model(df_selected_transformed, classifier)
        
        # x_selected_transformed = df_selected_transformed.iloc[:,:-1]
        # y_selected_transformed = df_selected_transformed.iloc[:,-1]
        # classifier.fit(x_selected_transformed,y_selected_transformed)

        # acc_score = accuracy_score(y_selected_transformed, 
        #                            classifier.predict(x_selected_transformed))
        five_acc_score_list.append(acc_score)
        print(f"Accuracy of the model with seed: {randSeed[i]} is: {acc_score}" )
        print('-'*80)
        print('-'*80)
    
    
    # # plot the mean and std, bar plot
    # for i in range(run_times):
    #     plt.bar(title+"_"+str(i+1), five_computional_time_dict["mean"][i], yerr=five_computional_time_dict["std"][i])
    #     plt.xlabel("Mean and Std for each run with different seeds")
    #     plt.ylabel("Time Spent (seconds)")
    #     plt.title(f"{title} \nmean and std of 5 computional time ")
    # plt.show()
    
    
    # for i in range(run_times):
    #     plt.bar(title+"_"+str(i+1), five_acc_score_list[i])
    #     plt.xlabel("Accuracy Score for each run with different seeds")
    #     plt.ylabel("Accuracy Score")
    #     plt.title(f"{title} :\n5 accuracy scores for each run on selected subsets")    
    # plt.show()
    
    # # plot generation vs fitness for each run
    # for i in range(run_times):
    #     plt.plot(logbook.select("gen"), logbook.select("best_fintess_current_gen"), 
    #              label=f"Seed: {str(randSeed[i])}")
    #     plt.legend(loc="lower right")
    #     # print(best_5_avg_fitness_list[i])
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.title(f"dataset: {title}\n 5 Curves for 5 runs ")
    # # for addressing the issue of log scale, which is
    # # very large and a very small fitness values in a plot
    # needSymlog = lambda y_values : min(y_values) < -1e-3
    # if needSymlog(logbook.select("best_fintess_current_gen")):
    #     plt.yscale('symlog') 
    # plt.show()
    
    # print the mean and std of acc and time spent on 5 runs
    print("-"*80)
    print(f"{title}")
    print(f"Mean of the accuracy score is: {np.mean(five_acc_score_list)} \
        \n  Std of the accuracy score is: {np.std(five_acc_score_list)}")
    print(f"Mean of the time spent is: {np.mean(five_computional_time_list)} \
        \n Std of the time spent is: {np.std(five_computional_time_list)}")
    
    five_acc_score_dict = {"mean":np.mean(five_acc_score_list), 
                           "std":np.std(five_acc_score_list),
                           "list":five_acc_score_list
                           }
    five_computional_time_dict = {"mean":np.mean(five_computional_time_list),
                                  "std":np.std(five_computional_time_list),
                                  "list":five_computional_time_list
                                  }
    return five_acc_score_dict, five_computional_time_list

# %%
# ds_sonar  = Sonar.constructFromFile("./sonar/sonar.data")
# ds_sonar.df_fs
# # ds_sonar.M

# %% [markdown]
# run filterGA 5 times for sonar and wbcd

# %%
ds_sonar  = Sonar.constructFromFile("./sonar/sonar.data")
sonar_filterGA_acc_score_dict, sonar_filterGA_computional_dict = \
    run_5_times_with_different_seeds(ds_sonar,
                                     "Sonar for FilterGA",
                                     evaluateFilterGA,
                                     classifier=CONSTANTS_DICT["CLASSIFIER"],
                                     max_gen=100,
                                     run_times=5)


# %%
print(f"acc_score for sonar filter GA: \n\t{sonar_filterGA_acc_score_dict}")
print("-"*40)
print(f"Time spent for sonar filter GA: \n\t{sonar_filterGA_computional_dict}")

# %%
ds_wbcd = Wbcd.constructFromFile("./wbcd/wbcd.data")
wbcd_filterGA_acc_score_dict, wbcd_filterGA_computional_dict = \
    run_5_times_with_different_seeds(ds_wbcd,
                                     "wbcd for FilterGA",
                                     evaluateFilterGA,
                                     classifier=CONSTANTS_DICT["CLASSIFIER"],
                                     max_gen=100,
                                     run_times=5
                                     )

# %%
print(f"acc_score for wbcd filter GA: \n\t{wbcd_filterGA_acc_score_dict}")
print("-"*40)
print(f"Time spent for wbcd filter GA: \n\t{wbcd_filterGA_computional_dict}")

# %% [markdown]
# run wrapperGA for 5 times

# %%
# https://machinelearningmastery.com/information-gain-and-mutual-information/
ds_sonar  = Sonar.constructFromFile("./sonar/sonar.data")
sonar_WrapperGA_acc_score_dict, sonar_WrapperGA_computional_dict = \
    run_5_times_with_different_seeds(ds_sonar,
                                     "Sonar for WrapperGA",
                                     evaluateWrapperGA,
                                     classifier=CONSTANTS_DICT["CLASSIFIER"],
                                     max_gen=100,
                                     run_times=5)


# %%
print(f"acc_score for sonar Wrapper GA: \n\t{sonar_WrapperGA_acc_score_dict}")
print("-"*40)
print(f"Time spent for sonar Wrapper GA: \n\t{sonar_WrapperGA_computional_dict}")

# %%
ds_wbcd = Wbcd.constructFromFile("./wbcd/wbcd.data")
wbcd_WrapperGA_acc_score_dict, wbcd_WrapperGA_computional_dict = \
    run_5_times_with_different_seeds(ds_wbcd,
                                     "wbcd for WrapperGA",
                                     evaluateWrapperGA,
                                     classifier=CONSTANTS_DICT["CLASSIFIER"],
                                     max_gen=100,
                                     run_times=5
                                     )

# %%
print(f"acc_score for wbcd Wrapper GA: \n\t{wbcd_WrapperGA_acc_score_dict}")
print("-"*40)
print(f"Time spent for wbcd Wrapper GA: \n\t{wbcd_WrapperGA_computional_dict}")


