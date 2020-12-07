import numpy as np
import pandas as pd 
from scipy.stats import truncnorm
import math
 

  

def tau(j,K):
  tau_1 = K + 1
  return tau_1 + int((j-1)/v) 


def strict_rate_UCB(S,g,K,v,T):
  """
  Returns which action to take: Pick player or run UCB

  Parameters
  ----------
  S : Indices for non-UCB arm selection
  g : Arms corresponding to indices in S
  K : Number of Arms
  v : Rate (must be less than 1/K)
  T : Time horizon

  Returns
  -------
  schedule of arms ,where 
  0,1,2,3 -> 0: UCB,  1: Player1, 2: Player 2, 3: Player 3
  """
  arms_selected = []  
  numbers_of_selections = [0] * K  #Keep track of number of times each arm gets selected
  sums_of_reward = [0] * K #Keep track of the total sum of rewards
  schedule = [] 
  # Algorithm initialization
  t = 1
  j = 1
  # Pull each arm onces
  while t <= K:
    i_t = t

    arms_selected.append(i_t)

    sums_of_reward[i_t-1] = df.values[t-1, i_t-1]
    numbers_of_selections[i_t-1] += 1
    schedule.append(i_t)

    t = t + 1
  # Main loop
  N = T-K # Number of times to iterate
  for j in range(1,int(N*v+1)):
    while t < tau(j+1,K):
      if t - tau(j,K) + 1 in S:
        i_t = int(g[t-tau(j,K)+1-1])
        schedule.append(i_t)
      else: 
        # UCB
        max_upper_bound = float('-inf')
        for i in range(0, K):
              #Confidence Interval
              conf_int = 2 * math.sqrt(math.log(T) / numbers_of_selections[i])
              
              #Average Reward
              average_reward = sums_of_reward[i] / numbers_of_selections[i]
              
              #Upper Bound
              upper_bound = (average_reward + conf_int)

              if upper_bound > max_upper_bound:
                  max_upper_bound = upper_bound
                  i_t = i+1
        schedule.append(0)
      arms_selected.append(i_t)
      arm_indx = int(i_t-1)
      numbers_of_selections[arm_indx] += 1
      sums_of_reward[arm_indx] += df.values[t-1, arm_indx]
      t = t + 1  
    j = j + 1
  return {'schedule':schedule, 'arms_selected':arms_selected, 'numbers_of_selections': numbers_of_selections}

def choose_g(S,v):
  g = np.zeros(int(1/v))
  k = 1
  for i in range(1,len(g)+1):
    if i in S:
      g[i-1] = k
      k = k + 1
  return g

def NormVar(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


K = 2  # Number of Arms
T = 102 # horizon including initialization of arms
v = 1.0/4 

# choose S and g for scheduling
S = [1,3]
g = choose_g(S,v)


#Create Data frame of Three Player Responses
p1=   NormVar(mean = .1, sd = .3, low = 0 , upp = 1  )
p2 =  NormVar(mean = .8, sd = .3, low = 0 , upp = 1  )
p3 =  NormVar(mean = .7, sd = .3, low = 0 , upp = 1  )

play1 = p1.rvs(1000)
play2 = p2.rvs(1000)
play3 = p3.rvs(1000)
data = {'p1': play1,'p2':play2,'p3':play3}
df = pd.DataFrame(data)


result = strict_rate_UCB(S,g,K,v,T)

print 'Schedule by AUCB algorithm; 0 indicates that the arm with the max UCB bound is chosen: '
print(result['schedule'])
print 'Actual sequence of arms: ' + str(result['arms_selected'])


print 'Number of selections per arm: ' + str(result['numbers_of_selections'])


print '\nExample schedules by AUCB algorithm for more combinations:'
K = 2
v = 1/3.0
T = 20
S = [2,3]
print '\nK = '+str(K)+', v = ' + str(v) + ', T = ' + str(T) +', S = ' + str(S)
g = choose_g(S,v)
result = strict_rate_UCB(S,g,K,v,T)
print(result['schedule'])

K = 3
v = 1/6.0
T = 20
S = [1,3,5]
print '\nK = '+str(K)+', v = ' + str(v) + ', T = ' + str(T)+', S = ' + str(S)
g = choose_g(S,v)
result = strict_rate_UCB(S,g,K,v,T)
print(result['schedule'])

K = 3
v = 1/4.0
T = 20
S = [1,2,3]
print '\nK = '+str(K)+', v = ' + str(v) + ', T = ' + str(T)+', S = ' + str(S)
g = choose_g(S,v)
result = strict_rate_UCB(S,g,K,v,T)
print(result['schedule'])
 

