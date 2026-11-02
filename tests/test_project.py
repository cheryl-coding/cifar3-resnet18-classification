"""Run using: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
from capstone import CLASSES, bootstrap_accuracy_ci, stratified_indices

class ProjectTests(unittest.TestCase):
    def test_counts_no_leakage(self):
        training = np.repeat([3, 5, 7], 800)
        testing = np.repeat([3, 5, 7], 200)
        records = stratified_indices(training, testing, seed=42)
        self.assertEqual(len(records), 2700)
        for label in CLASSES:
            for split, num in (("train", 600), ("val", 150), ("test", 150)):
                self.assertEqual(sum(r["class_name"] == label and r["split"] == split for r in records), num)
        pairs = [(r["source"], r["index"]) for r in records]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(r["source"] == "test" for r in records if r["split"] == "test"))
    def test_reproducibility(self):
        arr = np.repeat([3, 5, 7], 800)
        test = np.repeat([3, 5, 7], 200)
        self.assertEqual(stratified_indices(arr, test), stratified_indices(arr, test))
    def test_accuracy_ci(self):
        low, high = bootstrap_accuracy_ci([0,1,0,1], [0,1,0,1], n_boot=25)
        self.assertEqual((low, high), (1.0, 1.0))

if __name__ == "__main__":
    unittest.main()
