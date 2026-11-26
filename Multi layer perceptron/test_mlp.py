import pickle
import numpy as np

with open("model.pkl", "rb") as f:     # rb = read binary
    model = pickle.load(f)

x_test = np.array([
    [0.15, 0.75],  # 1
    [0.75, 0.15],  # 1
    [0.85, 0.65],  # 0
    [0.25, 0.35],  # 0
], dtype=float)

y_test = np.array([
    [1], [1], [0], [0],
], dtype=float)

for x,y in zip(x_test,y_test):
    output = 1 if (model.predict(x)) >= 0.5 else 0  
    print(f"Input {x} -> output {output} : Expected {y}")

