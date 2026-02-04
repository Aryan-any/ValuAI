import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.scanner_agent import ScannerAgent
from agents.deals import Deal, DealSelection

class TestScannerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ScannerAgent()

    def test_scanner_initialization(self):
        self.assertEqual(self.agent.name, "Scanner Agent")
        self.assertIsNotNone(self.agent.openai)

    @patch('agents.scanner_agent.ScrapedDeal.fetch')
    def test_fetch_deals_empty(self, mock_fetch):
        mock_fetch.return_value = []
        result = self.agent.fetch_deals(memory=[])
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
