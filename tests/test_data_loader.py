import unittest
from src.data_loader import load_weekly_data

class TestDataLoader(unittest.TestCase):
    def test_load_weekly_data(self):
        df = load_weekly_data()
        self.assertFalse(df.empty)
        self.assertIn('sales', df.columns)

if __name__ == "__main__":
    unittest.main() 