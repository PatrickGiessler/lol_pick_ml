import numpy as np
import json

class DataReader:
    def __init__(self, path: str):
        self.path = path

    def read_data(self):
        x_in = []
        y_in = []

        with open(self.path, "r") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    x_in.append(sample["input"])
                    y_in.append(sample["label"])
                except Exception as e:
                    print("Skipping invalid line:", e)
        y = np.array(y_in, dtype=np.float32)
        x = np.array(x_in, dtype=np.float32)
        return x, y    
                 
