# -*- coding: utf-8 -*-
"""Simulations_Using_MC

#Take A View At Different Numerical Simulations Approaches:::
    1.Black Scholes
  
    2.Binary Tree
  
    3.Monte Carlo


#Binary Tree Simulation :::::

"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

#model parameters
mu = 0.1      #mean
sigma = 0.15  #volatility
S0 = 1        #starting price
    
N = 10000     #number of simulations
T = [21.0/252, 1.0]  #time horizon in years
step = 1.0/252       #time step in years

#compute state price and probability
u = np.exp(sigma * np.sqrt(step))    #up state price
d = 1.0/u                            #down state price
p = 0.5+0.5*(mu/sigma)*np.sqrt(step) #prob of up state

"""#Binary Tree with Monte Carlo Simulation ::::
"""

#binomial tree simulation
up_times = np.zeros((N, len(T)))
down_times = np.zeros((N, len(T)))
for idx in range(len(T)):
    up_times[:,idx] = np.random.binomial(T[idx]/step, p, N)
    down_times[:,idx] = T[idx]/step - up_times[:,idx]

#compute terminal price
ST = S0 * u**up_times * d**down_times

#generate plots
plt.figure()
plt.plot(ST[:,0], color='b', alpha=0.5, label='1 month horizon')
plt.plot(ST[:,1], color='r', alpha=0.5, label='1 year horizon')
plt.xlabel('time step, day')
plt.ylabel('price')
plt.title('Binomial-Tree Stock Simulation')
plt.legend()
plt.show()

"""#MC Simulation has these two advantages over other simulations ::::

(1)No matter what the assets and the nature of their interaction you can completely strip of the mathematical construct of these assets and simply assuming a random normal path (geometric Brownian motion) simulate the forward movement of the assets

(2)Secondly, using a series of IF....THEN....ELSEIF....THEN statements we can simulate the payoff at termination.
"""

plt.figure()
plt.hist(ST[:,0], color='b', alpha=0.5, label='1 month horizon')
plt.hist(ST[:,1], color='r', alpha=0.5, label='1 year horizon')
plt.xlabel('price')
plt.ylabel('count')
plt.title('Binomial-Tree Stock Simulation')
plt.legend()
plt.show()

"""We can see from the above plots that our yearly estimates have higher volatility compared to the monthly estimates."""
