"""Sanity check published paired outcomes against the archived prediction files."""
import unittest
from pathlib import Path
from analyze_results import calculate

class MatchedResultTests(unittest.TestCase):
    def test_paired_predictions_and_counts(self):
        root = Path(__file__).resolve().parent.parent / 'runs'
        d = calculate(root)
        self.assertEqual(d['test_count'], 450)
        self.assertEqual((d['frozen_correct'],d['fine_tune_correct']), (387,409))
        self.assertEqual((d['frozen_wrong_fine_tune_correct'],d['frozen_correct_fine_tune_wrong']), (37,15))
        self.assertAlmostEqual(d['accuracy_difference'],22/450)
        self.assertGreater(d['accuracy_difference_95_ci_paired_bootstrap'][0],0)

if __name__=='__main__':unittest.main()
