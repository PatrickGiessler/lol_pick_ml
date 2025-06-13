import requests
import numpy as np

class DataFetcher:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def fetch_data(self):
        res = requests.get(self.endpoint)
        res.raise_for_status()
        data = res.json()["data"]
        x = np.array(data["x"], dtype=np.float32)
        y = np.array(data["y"], dtype=np.float32)
        return x, y
