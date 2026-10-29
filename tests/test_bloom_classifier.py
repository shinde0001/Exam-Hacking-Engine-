"""
Unit Tests for Bloom's Taxonomy Classifier (engine/bloom_classifier.py)
"""

import unittest
from engine import bloom_classifier


class TestBloomClassifier(unittest.TestCase):
    def test_classification_levels(self):
        """Verify keyword classification across all 6 cognitive tiers."""
        # Level 1: Remember
        res_rem = bloom_classifier.classify_bloom_level("Define Newton's first law of motion")
        self.assertEqual(res_rem.level, "remember")
        self.assertEqual(res_rem.tier, 1)

        # Level 2: Understand
        res_und = bloom_classifier.classify_bloom_level("Explain the mechanism of enzyme catalysis")
        self.assertEqual(res_und.level, "understand")
        self.assertEqual(res_und.tier, 2)

        # Level 3: Apply
        res_app = bloom_classifier.classify_bloom_level("Calculate the acceleration of a 5kg mass")
        self.assertEqual(res_app.level, "apply")
        self.assertEqual(res_app.tier, 3)

        # Level 4: Analyze
        res_ana = bloom_classifier.classify_bloom_level("Compare and contrast mitosis and meiosis")
        self.assertEqual(res_ana.level, "analyze")
        self.assertEqual(res_ana.tier, 4)

        # Level 5: Evaluate
        res_eva = bloom_classifier.classify_bloom_level("Justify the use of renewable energy over fossil fuels")
        self.assertEqual(res_eva.level, "evaluate")
        self.assertEqual(res_eva.tier, 5)

        # Level 6: Create
        res_cre = bloom_classifier.classify_bloom_level("Derive the expression for capacitance of a parallel plate capacitor")
        self.assertEqual(res_cre.level, "create")
        self.assertEqual(res_cre.tier, 6)

    def test_compute_distribution(self):
        """Verify corpus distribution calculation."""
        questions = [
            {"question_text": "Define velocity and acceleration"},
            {"question_text": "Calculate momentum of moving vehicle"},
            {"question_text": "Explain Doppler effect in sound"},
            {"question_text": "Derive lens formula"},
        ]
        dist = bloom_classifier.compute_bloom_distribution(questions)
        self.assertEqual(dist["total_questions"], 4)
        self.assertGreater(dist["average_cognitive_tier"], 1.0)
        self.assertIn("percentages", dist)
        self.assertEqual(sum(dist["counts"].values()), 4)


if __name__ == "__main__":
    unittest.main()
