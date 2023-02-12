from recode.utils import output_adj
from recode.rl import RL
import pandas as pd
import numpy as np
from recode.causal_strength import adj_cs

# 导入数据
X = pd.read_csv("C:/Users/14903/Desktop/NHANES/2011-2020/标准化/2011-2020_n_0.csv")
X = np.array(X)

# 训练学习
rl = RL(nb_epoch=2000, device_type="gpu", score_type="BIC")
rl.learn(X)
print(rl.causal_matrix)
print(adj_cs(rl.causal_matrix, X))
# 显示图和计算指标
output_adj(rl.causal_matrix)
