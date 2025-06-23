import unittest
import numpy as np
from src.model import train_model

class TestModel(unittest.TestCase):
    def test_train_model(self):
        # Dummy data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        model, preds = train_model(X, y)
        self.assertEqual(len(preds), 20)  # 20% test split

if __name__ == "__main__":
    unittest.main() 