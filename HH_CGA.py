#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt 
import seaborn as sns
import time
from pathlib import Path


# In[88]:


#bin Packaging 1d
data_folder = Path(r"C:\Users\monit\Documents\Tec\CGA\BPP instances\Irnich_BPP")
file="csAA125_1.txt"
file_to_open = data_folder / file
File = open(file_to_open,'r').readlines()
number_bins=int(File[0])
bin_size=int(File[1])
number_items=len(File)-2
items=np.zeros((1,number_items))
for x in range(2,len(File)):
    items[0,x-2]=int(File[x])
print(items.shape,bin_size)


# In[43]:


#bin packaging creation
def population_creation(rules,fs,d1,d2):
    population=np.zeros((d1,d2,((rules*fs)+rules)))
    for k in range(np.shape(population)[0]):
        for z in range(np.shape(population)[1]):
            population[k,z,:]=np.random.randint(low=0, high=1000,size=population.shape[2])      
    return population         


# In[7]:


def crossover(fitness,population,d1,d2,number_items,bin_size,mr):
    #this function choses the best fitness of the neighbors and then convines both parents, if the solution has better fitness 
    #and it is viable it will replace the individual being tested
    new_population=np.zeros(np.shape(population))
    assert new_population.shape == population.shape
    for k in range(np.shape(population)[0]): 
        for z in range(np.shape(population)[1]):
            parent_1=population[k,z,:].copy()
            #l5 neighbors
            l=np.array([fitness[k,(z-1)%d2],fitness[(k-1)%d1,z],fitness[k,(z+1)%d2],fitness[(k+1)%d1,z]]) 
            selector=np.random.choice(np.arange(4), size=2, replace=False)
            s_v=np.array([selector[0],selector[1]])
            fit=[l[selector[0]],l[selector[1]]]
            index=np.where(fit==np.max(fit))
            index=list(zip(*index))[0]
            winner=s_v[index]
            #print(parent_1,"parent1")
            if winner==0:
                parent_2=population[k,(z-1)%d2,:].copy()
            elif winner==1:    
                parent_2=population[(k-1)%d1,z,:].copy()
            elif winner==2:
                parent_2=population[k,(z+1)%d2,:].copy()
            elif winner==3:
                parent_2=population[(k+1)%d1,z,:].copy()                
            prepop=offspring_op(parent_2,parent_1,population)  
            #print(parent_1,parent_2,prepop)
            prepop=mutation(prepop,mr)
            
            new_population[k,z,:]=prepop
    
            #print(new_population[k,z,:],"fianl")  
            #print("----------------------")              
    return new_population


# In[8]:


#convines the parents 
def offspring_op(parent_2,parent_1,population):
    rnd=np.random.uniform(low=0, high=1)
    max_crosover =np.uint8(np.shape(population)[2])
    crossover_point=np.random.randint(low=1,high=max_crosover-1)
    if rnd >.5:
        offspring = parent_1[0:crossover_point]
        offspring = np.concatenate((offspring,parent_2[crossover_point:]),0)
    else:
        offspring = parent_2[0:crossover_point]
        offspring = np.concatenate((offspring,parent_1[crossover_point:]),0)    
    return offspring


# In[9]:


#mutates a random gene with a random bin if the random number generator is below or equal to the mutation rate, then checks 
#if that solution is posible, if it is the gene is changed otherwise it reamins the same
def mutation(population,mr): 
    if np.random.uniform(low=0, high=1)<=mr:
                #print(k,z,"mutation")
                # The random bin to be changed in the genome.
        random_number = np.random.randint(low=0, high=1000)
        cromo = randint(0, np.shape(population)[0]-1)               
        gene=population.copy()               
        gene[cromo]=random_number              
        population=gene                                                     
    return population


# In[74]:


def bin_heuristic(population,items,rules,bin_size): 
    bins_used=[]
    bins=np.array([0])
    bins_capacity=np.array([0])
    lastitem=0
    for x in range (items.shape[1]):
        #print(x,"x")
        items_copy=items[0,x:]
        #print(items_copy,"items")
        fnormal=features_def(rules,items_copy,bins_capacity[bins_capacity.shape[0]-1],abs(items[0,x]-lastitem+.0001)) #features
        euc_distance=np.zeros((1,rules))
        count=int(fnormal.shape[0]/rules)
        population_for_euc=[]
        #temporary vector for easy eucl rest
        #print(population)
        for z in range(population.shape[0]):
            #print(count,"count")
            if (count!=0):
                population_for_euc.append(population[z]/1000)
                count=count-1
                #print(population_for_euc,"pop for euc")
            else:
                count=int(fnormal.shape[0]/rules)
        population_for_euc=np.array(population_for_euc)
        #print(population_for_euc,"pop_for euc")
        #print(fnormal,"fnormal")
        o=0
        for i in range (rules):
            
            pre_euc=(fnormal-population_for_euc)**2
            #print(pre_euc,"pre_euc")
            #print(pre_euc[o:o+int(fnormal.shape[0]/rules)],"el que tomas")
            euc_distance[0,i]=np.sqrt(np.sum(pre_euc[o:o+int(fnormal.shape[0]/rules)]))
            o=o+int(fnormal.shape[0]/rules)
            #print(i,"i")
            #print(o,"0")
        #print(population,"population")
        #print(euc_distance,"euc dist")
        best=np.max(-1*euc_distance)
        #print(best,"minimo")
        index=np.where(euc_distance==(best*-1))
        index=list(zip(*index))[0]
        index=list(index)
        #print(index,"index")
        #print(((index[1]+1)*int(fnormal.shape[0]/rules))+index[1])
        #features.shape[0]/rules ??? no recuerdo esto
        heu_select=population[((index[1]+1)*int(fnormal.shape[0]/rules))+index[1]]
        #-------------------------------------------
        if(heu_select<=250):
            #print("first fit")
            new=1
            for m in range (bins.shape[0]):
                if (bins_capacity[m]+items[0,x]<=bin_size):
                    bins_used.append(m)
                    bins_capacity[m]=(bins_capacity[m]+items[0,x])
                    new=0
                    break
                    
            if new==1:
                bins_used.append(bins.shape[0])
                bins=np.append(bins,bins.shape[0])
                bins_capacity=np.append(bins_capacity,items[0,x])
           
        elif heu_select<=500:
            #print("best fit")
            best_fit=np.zeros((bins.shape[0]))
            conta=0
            for m in range(bins.shape[0]):
                best_fit[m]=bin_size-bins_capacity[m]-items[0,x]
                if best_fit[m]<0:
                    conta=conta+1
                    best_fit[m]=np.max(items)*100
            if conta==bins.shape[0]:
                bins_used.append(bins.shape[0])
                bins=np.append(bins,bins.shape[0])
                bins_capacity=np.append(bins_capacity,items[0,x])
            
            else:
                index1=np.where(-1*best_fit==np.max(-1*best_fit))
                index1=list(zip(*index1))[0]
                index1=list(index1)
                #print(index1)
                bins_used.append(bins[index1[0]])
                bins_capacity[index1[0]]=bins_capacity[index1[0]]+items[0,x]
                
        elif(heu_select<=750):
            #print( "#worst fit")
            best_fit=np.zeros((bins.shape[0]))
            conta=0
            for m in range(bins.shape[0]):
                best_fit[m]=bin_size-bins_capacity[m]-items[0,x]
                if best_fit[m]<0:
                    conta=conta+1
            if conta==bins.shape[0]:
                bins_used.append(bins.shape[0])
                bins=np.append(bins,bins.shape[0])
                bins_capacity=np.append(bins_capacity,items[0,x])
            
            else:
                index1=np.where(best_fit==np.max(best_fit))
                index1=list(zip(*index1))[0]
                index1=list(index1)
                #print(index1)
                bins_used.append(bins[index1[0]])
                bins_capacity[index1[0]]=bins_capacity[index1[0]]+items[0,x]
            
        elif(heu_select<=1000):
            #print("#second worst fit")
            best_fit=np.zeros((bins.shape[0]))
            conta=0
            best_fit_posit=[]
            best_fit_posti=np.array(best_fit_posit)
            for m in range(bins.shape[0]):
                best_fit[m]=bin_size-bins_capacity[m]-items[0,x]
                if best_fit[m]<0:
                    conta=conta+1
                else:
                    best_fit_posit=np.append(best_fit_posit,best_fit[m])                    
            if conta==bins.shape[0]:
                bins_used.append(bins.shape[0])
                bins=np.append(bins,bins.shape[0])
                bins_capacity=np.append(bins_capacity,items[0,x])
            
            else:
                best_sorted=np.sort(best_fit_posit)
                index1=np.where(best_fit==best_sorted[(best_fit_posit.shape[0])-2])
                index1=list(zip(*index1))[0]
                index1=list(index1)
                bins_used.append(bins[index1[0]])
                bins_capacity[index1[0]]=bins_capacity[index1[0]]+items[0,x] 
        lastitem=items[0,x]
        #print(bins,"bins")
        #print(bins_capacity,"bins capacity")
        #print(bins_used,"bins used")
    bins_used=np.array(bins_used)
    #print(bins_used,"after np")
    fitness=cal_pop_fitness_unit(bins_used,items,bin_size)
    return fitness


# In[84]:


def cal_pop_fitness_unit(population,items,bin_size):
    #print(population,items)
    bins_used=np.unique(population)
    fi=np.zeros((1,np.shape(bins_used)[0]))
    for o in range(np.shape(bins_used)[0]):   
        index3=np.where(population==bins_used[o])
        #print(index3)
        count=np.sum(items[0,index3[0]])
        #print(count)
        if count>bin_size:
            fi[0,o]=0
        else:
            fi[0,o]=(count/bin_size)**2
    fitness=np.sum(fi)/np.shape(bins_used)[0]
    return fitness


# In[81]:


def features_def(rules,items,capacity,diff):
    f1=np.average(items)
    f2=np.std(items)
    #f3=np.unique(items).shape[0]
    f4=capacity
    f5=diff
    features=[]
    for l in range(rules):
        features.append(f1)
        features.append(f2)
        #features.append(f3)
        features.append(f4)
        features.append(f5)
    features=np.array(features)
    fnormal=features/np.max(features)
    return fnormal


# In[89]:


#CGA
d1=16
d2=16
rules=6
fs=int(features_def(rules,items,0,0).shape[0]/rules)
#print(fs)
mr=.05
generations=11
population=population_creation(rules,fs,d1,d2)
#print(population.shape)
start_time = time.time()
x1=[]
y1=[]
for l in range(generations):
    fitness=np.zeros((d1,d2))
    for k in range (d1):
        for z in range (d2):
            fitness[k,z]=bin_heuristic(population[k,z,:],items,rules,bin_size)
    #print(fitness)
    print(np.max(fitness))
    new_population=crossover(fitness,population,d1,d2,number_items,bin_size,mr)
    population=new_population
    x1.append(l)
    y1.append(np.max(fitness))
"-----------------------------------------------------------------"    
elapsed_time2= time.time() - start_time
total_time=(elapsed_time2)*1000

best=np.max(fitness)

print("time algorith",elapsed_time2)
print("Best result fianl : ", best)
plt.plot(x1, y1, label = "fitness equation") 
plt.xlabel('generations')
plt.ylabel('max fitness')
plt.show() 
print("----------------")
print("distribution of the fitness in the matrix")
p1 = sns.heatmap(fitness)


# In[ ]:




