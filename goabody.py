def GOA(N, Max_iter, lb, ub, dim, fobj):
  grasshopperPositions = initialization(N, dim, ub, lb)
  targetPosition = np.copy(min(grasshopperPositions,  key=(lambda x: fobj(x))))
  
  cMax = 1.0
  cMin = 0.000001
  
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
      
    bestNewPosition = min(grasshopperPositions,  key=(lambda x: fobj(x)))
    if fobj(bestNewPosition) < fobj(targetPosition):
      targetPosition = np.copy(bestNewPosition)
    populationStats(l, grasshopperPositions, targetPosition, fobj)
  return stats
