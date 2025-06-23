import unittest
from src.data_processor import process_weekly_data

class TestDataProcessor(unittest.TestCase):
    def test_process_weekly_data(self):
        df = process_weekly_data()
        self.assertFalse(df.empty)
        self.assertIn('creative_text', df.columns)
        self.assertIn('creative_length', df.columns)

if __name__ == "__main__":
    unittest.main() 