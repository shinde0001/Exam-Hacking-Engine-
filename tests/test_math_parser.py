"""
Unit Tests for Math Parser (engine/math_parser.py)
"""

import unittest
from engine import math_parser


class TestMathParser(unittest.TestCase):
    def test_contains_math(self):
        """Verify mathematical expression detection."""
        self.assertTrue(math_parser.contains_math("Calculate the value of x² + 2x + 1 = 0"))
        self.assertTrue(math_parser.contains_math("Evaluate $E = mc^2$ in vacuum"))
        self.assertTrue(math_parser.contains_math(r"Find \frac{dy}{dx} if y = \sin(x)"))
        self.assertTrue(math_parser.contains_math("Compute √16 + 5"))
        self.assertTrue(math_parser.contains_math("If 2x + 3 = 7, find x"))
        self.assertFalse(math_parser.contains_math("Define Newton's first law of motion in English"))
        self.assertFalse(math_parser.contains_math("List three factors affecting soil erosion"))

    def test_normalize_superscripts(self):
        """Verify unicode exponent normalization."""
        self.assertEqual(math_parser.normalize_superscripts("x²"), "x^2")
        self.assertEqual(math_parser.normalize_superscripts("y³ + z⁴"), "y^3 + z^4")
        self.assertEqual(math_parser.normalize_superscripts("aⁿ"), "a^n")

    def test_normalize_math_expression(self):
        """Verify canonical representation of mathematical strings."""
        # Unicode superscripts and LaTeX symbols
        expr = r"$\frac{x}{2} + x² \times 3$"
        norm = math_parser.normalize_math_expression(expr)
        self.assertIn("((x) / (2))", norm)
        self.assertIn("x ^ 2", norm)
        self.assertIn("*", norm)

        # Division & sqrt
        expr2 = r"\sqrt{y} \div 4"
        norm2 = math_parser.normalize_math_expression(expr2)
        self.assertIn("((y) ^ (1 / 2))", norm2)
        self.assertIn("/", norm2)

    def test_algebraic_equivalence(self):
        """Verify SymPy structural and algebraic equivalence."""
        # (x + 1)^2 == x^2 + 2x + 1
        self.assertTrue(
            math_parser.are_math_expressions_equivalent(
                "(x + 1)^2",
                "x^2 + 2*x + 1",
            )
        )

        # Trigonometric identity: sin^2(x) + cos^2(x) == 1
        self.assertTrue(
            math_parser.are_math_expressions_equivalent(
                "sin(x)**2 + cos(x)**2",
                "1",
            )
        )

        # Equation equivalence: 2*x = 4 equivalent to x = 2
        self.assertTrue(
            math_parser.are_math_expressions_equivalent(
                "2*x = 4",
                "x = 2",
            )
        )

        # Clearly non-equivalent expressions
        self.assertFalse(
            math_parser.are_math_expressions_equivalent(
                "x + 1",
                "x + 2",
            )
        )


if __name__ == "__main__":
    unittest.main()
