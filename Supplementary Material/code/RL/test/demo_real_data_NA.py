from recode.utils import output_adj
from recode.rl import RL
import pandas as pd
import numpy as np
from recode.causal_strength import adj_cs


X = pd.read_csv("C:/Users/14903/Desktop/NHANES/2011-2020/2011-2020.csv")
X = np.array(X)

rl = RL(nb_epoch=2000, device_type="gpu", score_type="BIC")
rl.learn(X)
print(rl.causal_matrix)
print(adj_cs(rl.causal_matrix, X))

output_adj(rl.causal_matrix)
