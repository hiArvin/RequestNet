import numpy as np
sp = np.array([[1, 0, 3],
[4, 5, 6],
[7, 0, 9]])
mask = sp!=0

print((mask))
print(np.sum(mask,axis=1))