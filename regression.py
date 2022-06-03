import numpy as np
import pandas as pd
import random


# Parsing data
data=pd.read_csv("Data/math_data.csv")
usable = np.array(data[["Bias", "Fairway%", "GIR%", "Avg. Putts", "Scarmbling %"]])

# X = np.hstack(usable)
X = pd.DataFrame(usable).to_numpy().tolist()
y = pd.DataFrame(np.array(data[["Avg. Score"]])).to_numpy()

y_2 = []

for a in y:
    y_2.append(a[0])

y = np.array(y_2)
y = y.astype('int')
X = np.array(X)

# transpsoe x
X_t = np.transpose(X)

theta = np.array([random.random() for i in range(5)])

# theta = np.array([55.0, 0.03, -0.2, 1.0, -0.02])

def gradient(theta):
    # (Xt * X) * theta   -   Xt * y
    return np.subtract(np.matmul(np.matmul(X_t, X), theta), np.matmul(X_t, y)) 

# gradient(theta)


# gradient descent
l_rate = np.array([0.0000001 for i in range(5)])
def gradeint_descent(gradient, start, learning_rate, n_iter=2000, tolerance=1e-06):
    vector = start
    
    for _ in range(n_iter):
        diff = np.matmul(-learning_rate, gradient(vector))
        if np.all(np.abs(diff) <= tolerance):
            break
            
        vector = np.add(vector, diff)
#         print(vector)
        
    return vector

decent = gradeint_descent(gradient, theta, l_rate)

for num in decent:
    print("{:.4f}".format(float(num)))
