# -*- coding: utf-8 -*-
"""Monte_Carlo.ipynb
"""

# example of effect of size on monte carlo sample
from numpy.random import normal
from matplotlib import pyplot
# define the distribution
mu = 50
sigma = 5
# generate monte carlo samples of differing size
sizes = [10, 50, 100, 1000]
for i in range(len(sizes)):
	# generate sample
	sample = normal(mu, sigma, sizes[i])
	# plot histogram of sample
	pyplot.subplot(2, 2, i+1)
	pyplot.hist(sample, bins=20)
	pyplot.title('%d samples' % sizes[i])
	pyplot.xticks([])
# show the plot
pyplot.show()

"""
#Monte Carlo Integration For Finding Value Of PI ::::

For every single value of 10,100,1000,10000 and finally at 100000 ,1000000 .....
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
#Defining the lines of the square:
horiz = np.array(range(100))/100.0
y_1 = np.ones(100)
plt.plot(horiz , y_1, 'b')
vert = np.array(range(100))/100.0
x_1 = np.ones(100)
plt.plot(x_1 , vert, 'b')
#Plotting the random points:
import random
inside = 0
i=1
n=int(input("Enter the total number of points: "))
while (i<=n):
  x = random.random()
  y = random.random()
  if ((x**2)+(y**2))<=1:
    inside+=1
    plt.plot(x , y , 'go')
  else:
    plt.plot(x , y , 'ro')
  i+=1
pi=(4*inside)/n
print ("The value of pi is:")
print(pi)
plt.show()
plt.savefig("Piii")

'''
-->>
'''
import numpy as np
import matplotlib.pyplot as plt
n = 1000000
x = np.random.rand(n,2)
inside =x[np.sqrt(x[:,0]**2+x[:,1]**2) < 1]
estimate = 4*len(inside)/len(x)
print("Estimate of pi: {}".format(estimate)) 
plt. figure(figsize=(8,8)) 
plt.scatter(x[:,0], x[:,1],s= 0.5, c='red')
plt.scatter(inside[:,0], inside[ :,1], s=0.5 ,c='blue')
plt.show()

