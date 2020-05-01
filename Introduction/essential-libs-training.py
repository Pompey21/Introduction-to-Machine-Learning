import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

x = np.linspace(-10,10,100)
y = np.sin(x)

# plt.plot(x, y, marker="x")
# plt.show()



data = {'Name' : ["John","Anna","Peter","Linda"],
		'Location' : ["New York", "Paris", "Berlin", "London"],
		'Age' : [24,13,53,33]
		}

data_pandas = pd.DataFrame(data)
print(data_pandas[data_pandas.Age>30])