# MCMC-Algorithm-Implementation

### Markov Chain Monte Carlo(MCMC)-A Machine Learning Algorithm for posterior distribution

MCMC methods are used to approximate the posterior distribution of a parameter of interest by random sampling in a probabilistic space.

With MCMC,i.e.Markov Chain + Monte Carlo Algorithms, we draw samples from a (simple) proposal distribution so that each draw depends only on the state of the previous draw (i.e. the samples form a Markov chain).

Under certain condiitons, the Markov chain will have a unique stationary distribution. In addition, not all samples are used -

instead we set up acceptance criteria for each draw based on comparing successive states with respect to a target distribution that enusre that the stationary distribution is the posterior distribution of interest.

The nice thing is that this target distribution only needs to be proportional to the posterior distribution, which means we don’t need to evaluate the potentially intractable marginal likelihood, which is just a normalizing constant..

#
#
#### The repository contains FOUR different modules ::::::

#
Let's first have brief about MONTE CARLO ::::
    
1.Monte_Carlo ::::

Basic Implementation Of Monte Carlo Method -

    -->> For Sampling probability distribution
    -->> For estimating value of PI
    
#
2.Random_Walk_MC ::::

Random Walk Monte Carlo Implementation On 2D Lattice -

    -->> With small difference number of steps and iterations 
    -->> With large difference number of steps and iterations 
    -->> With large number of iterations
    -->> Steps are smaller than iterations

#
3.Simulation_Using_MC :::::

Numerical Simuation Variants - 

    -->> Binary Tree Simulation
    -->> Black Scholes Simulation
    -->> Monte Carlo Simulation 

#
MCMC is an advancement over MONTE CARLO ::::

4.MCMC_ALGORITHM :::::

MCMC And It's Common Variants -
    
    -->> Bayesian Algorithm
    -->> Metropolis-Hastings Sampler Algorithm
    -->> Gibbs Sampler Algorithm 
    -->> slice sampling Sampler Algorithm
    -->> Hierarchical Method
    
#
#     
##### Now MCMC thus can be applied for a variety of optimization problems namely:
1.	Finding out the best arrangement of a DNA sequence
2.	Cryptography: Breaking the code.
3.	Heuristic way of Minimizing/Maximizing any function
4.	Sampling from any distribution
5.	Industrial Engineering and Operations Research
6.	Random Graphs and Combinatorial Structures
7.	Economics and Finance
8.	Computational Statistics and much more...... !!!!!

#
