import unittest
import numpy as np
from Scripts.resume_ranking import normalize_scores, get_score_label

class TestNLPLogic(unittest.TestCase):
    def test_score_normalization_equal(self):
        # Scores should map directly to absolute percentages
        scores = [0.8, 0.8, 0.8]
        normalized = normalize_scores(scores)
        np.testing.assert_array_almost_equal(normalized, [80.0, 80.0, 80.0])

    def test_score_normalization_range(self):
        # Scores should map directly to absolute percentages
        scores = [0.1, 0.5, 0.9]
        normalized = normalize_scores(scores)
        np.testing.assert_array_almost_equal(normalized, [10.0, 50.0, 90.0])

    def test_score_labels(self):
        self.assertEqual(get_score_label(90), "🟢 Excellent Match")
        self.assertEqual(get_score_label(70), "🔵 Good Match")
        self.assertEqual(get_score_label(50), "🟡 Average Match")
        self.assertEqual(get_score_label(30), "🟠 Weak Match")
        self.assertEqual(get_score_label(10), "🔴 Poor Match")

if __name__ == '__main__':
    unittest.main()
