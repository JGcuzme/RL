from recode.utils import output_adj
from recode.rl import RL
import pandas as pd
import numpy as np
from recode.causal_strength import adj_cs


X = pd.read_csv("C:/Users/14903/Desktop/data/2000/原始数据集/2000.csv")
X = np.array(X.drop(columns="Outcome00"))
# X = np.array(X.drop(columns="Outcome"))   

rl = RL(nb_epoch=20000, device_type="gpu", score_type="BIC")
rl.learn(X)
print(rl.causal_matrix)
print(adj_cs(rl.causal_matrix, X))

output_adj(rl.causal_matrix)
