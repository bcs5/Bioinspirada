import numpy as np
import copy

def initialization(N, dim, up, down):
  return np.multiply(np.random.rand(N, dim), (up-down)) + down

def distance (a, b):
  return np.linalg.norm(a-b)

def S_func(r):
  f = 0.5;
  l = 1.5;
  return f * np.exp(-r/l) - np.exp(-r)
  
def decide(probability):
  return np.random.rand() < probability

def rem (a, b):
  return a - b*np.floor(a/b)

def ackley (xs, c1 = 20, c2 = 0.2, c3 = 2*np.pi):
  Dim = len(xs)
  return - c1 * np.exp(-c2 * np.sqrt(np.sum(xs*xs)/Dim)) - np.exp( np.sum(np.cos(np.multiply(xs, c3))/Dim)) + c1 + np.exp(1)

def nicheRepulsing (population, fobj, lb, ub):
  N = len(population)
  for i in range (N):
    rep = population[i]
    niche = [rep]
    
    tot_dist = 0.0
    for j in range(N):
      if (i == j):
        continue
      tot_dist += distance(population[i], population[j])
    avg_dist = tot_dist / N
    
    for j in range(i+1, N):
      dist = distance(rep, population[j])
      if (dist < avg_dist):
        niche.append(population[j])
    
    best = min(niche,  key=(lambda x: fobj(x)))
    worst = max(niche,  key=(lambda x: fobj(x)))
    population[i] = best
    
    for j in range(i+1, N):
      if (i == j):
        continue
      dist = distance(rep, population[j])
      if (dist < avg_dist):
        x1 = np.random.rand()
        x2 = np.random.rand()
        population[j] = np.clip(np.multiply(best, x1) + np.multiply(worst, x2), lb, ub)
  return population
  
def bhc (grasshopper, beta, lb, ub):
  dim = grasshopper.shape[0]
  return (ub+lb) - grasshopper if (decide(beta)) else np.random.rand(dim) * (ub -lb) + lb

def populationStats (iteration, population, targetPosition, fobj):
  fitnessArr = [fobj(x) for x in population]
  avg = np.average(fitnessArr)
  std = np.std(fitnessArr)
  return [iteration, fobj(targetPosition), avg, std]
  
watchedExecutions = list(range(0, 10)) + [10, 100, 500]

def GOA(N, Max_iter, lb, ub, dim, fobj, useBhc = True, useNicheRepulsing = True):
  grasshopperPositions = initialization(N, dim, ub, lb)
  
  targetPosition = np.copy(min(grasshopperPositions,  key=(lambda x: fobj(x))))
  
  cMax = 1.0
  cMin = 0.000001
  beta = 0.5
  
  stats = [populationStats(0, grasshopperPositions, targetPosition, fobj)]
  for l in range(1, Max_iter+1):
    c = cMax - l * ((cMax-cMin)/Max_iter)
    temp = np.copy(grasshopperPositions)
    for i in range(N):
      S_i = np.zeros(dim);
      for j in range(N):
        if (i == j):
          continue
        dist = distance(temp[i], temp[j])
        r_ij_vec = (temp[j] - temp[i]) / (dist + 1e-14)
        xj_xi = 2+rem(dist, 2)
        s_ij = np.multiply(((ub - lb)*c/2)*S_func(xj_xi), r_ij_vec);
        S_i += s_ij;
      grasshopperPositions[i] = np.clip(np.multiply(c, S_i) + targetPosition, lb, ub)
      
      
      if (useBhc and fobj(grasshopperPositions[i]) < fobj(targetPosition)):
        bhcGrasshopper = bhc(grasshopperPositions[i], beta, lb, ub)
        if (fobj(bhcGrasshopper) <= fobj(grasshopperPositions[i])):
          grasshopperPositions[i] = bhcGrasshopper
          
    if (useNicheRepulsing):
      grasshopperPositions = nicheRepulsing(grasshopperPositions, fobj, lb, ub)
    bestNewPosition = min(grasshopperPositions,  key=(lambda x: fobj(x)))
    if fobj(bestNewPosition) < fobj(targetPosition):
      targetPosition = np.copy(bestNewPosition)
    
    if (l in watchedExecutions):
      stats += [populationStats(l, grasshopperPositions, targetPosition, fobj)]
  return stats

executionsStats = None
NEXECS = 30
for i in range (NEXECS):
  stats = GOA(30, 500, -5.0, 5.0, 30, ackley)
  if executionsStats is None:
    executionsStats = stats
  else:
    for j in range(len(executionsStats)):
      for k in range(1, 4):
        executionsStats[j][k] += stats[j][k]

for j in range(len(executionsStats)):
  for k in range(1, 4):
    executionsStats[j][k] /= NEXECS

print (executionsStats)
