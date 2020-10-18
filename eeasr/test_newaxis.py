import numpy as np
import random
random.seed(0)
x = np.random.randint(1, 8, size=5)
x1 = x[np.newaxis, :]
y = np.random.randint(1, 8, size=5)
y1 = y[np.newaxis, :]
import pdb;pdb.set_trace()
new = np.concatenate((x1,y1),0)
sample = random.sample(range(10), 2)
sample1 = random.sample(range(10), 2)
a = np.random.random_sample()
b = np.random.random_sample()
print('end')
