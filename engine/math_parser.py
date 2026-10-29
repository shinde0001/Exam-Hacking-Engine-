"""
Exam-Hacking Engine — Math Parser
Handles detection, normalization, and structural equivalence of mathematical expressions.
Features:
- Regex detection of mathematical notation (operators, superscripts, fractions, Greek letters, LaTeX)
- Canonical normalization (e.g., x^2 == x² == x**2, unicode exponents to ASCII)
- SymPy integration for structural & algebraic equivalence checking
"""

import re

try:
    import sympy
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    HAS_SYMPY = True
    SYMPY_TRANSFORMATIONS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
except ImportError:
    HAS_SYMPY = False
    SYMPY_TRANSFORMATIONS = ()

# Unicode superscripts to ASCII
UNICODE_SUPERSCRIPTS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")",
    "ⁿ": "n", "ⁱ": "i",
}

# Common Greek symbols
GREEK_LETTERS = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "θ": "theta", "λ": "lambda", "μ": "mu", "π": "pi",
    "σ": "sigma", "ω": "omega", "φ": "phi", "Δ": "Delta",
}

# Regex to detect math indicators
MATH_INDICATOR_PATTERN = re.compile(
    r"(?:"
    r"[\^²³√∫∑∏±×÷≠≤≥≈∞]"                          # Math symbols
    r"|\b(?:sin|cos|tan|cot|sec|csc|log|ln|lim)\b"    # Common math functions
    r"|\$[^$]+\$"                                   # Inline LaTeX $...$
    r"|\\frac\{[^{}]+\}\{[^{}]+\}"                  # LaTeX fractions
    r"|\\sqrt\{[^{}]+\}"                            # LaTeX sqrt
    r"|[a-zA-Z]\s*[\+\-\*\/\^=]\s*[0-9a-zA-Z]"      # Algebraic expression like x + 2, y = mx
    r"|\b\d+\s*[\+\-\*\/]\s*\d+\b"                  # Arithmetic like 5 + 3
    r"|[0-9a-zA-Z]+\s*=\s*[0-9a-zA-Z]+"             # Equations like F = ma, E = mc^2
    r")",
    re.IGNORECASE,
)


def contains_math(text: str) -> bool:
    """
    Check if a text snippet contains mathematical expressions, symbols, or formulas.
    """
    if not text:
        return False
    return bool(MATH_INDICATOR_PATTERN.search(text))


def normalize_superscripts(text: str) -> str:
    """
    Convert unicode superscripts to standard '^' exponent form.
    Example: 'x² + y³' -> 'x^2 + y^3'
    """
    if not text:
        return ""
    result = []
    in_super = False
    for ch in text:
        if ch in UNICODE_SUPERSCRIPTS:
            if not in_super:
                result.append("^")
                in_super = True
            result.append(UNICODE_SUPERSCRIPTS[ch])
        else:
            in_super = False
            result.append(ch)
    return "".join(result)


def normalize_math_expression(expr_str: str) -> str:
    """
    Normalize math text into canonical syntax for comparison:
    - Converts unicode superscripts (x² -> x^2)
    - Normalizes exponents (x**2 -> x^2)
    - Normalizes multiplication symbols (×, \\times, \\cdot -> *)
    - Normalizes division (÷, \\div -> /)
    - Strips LaTeX math delimiters ($...$, \\[...\\])
    - Normalizes LaTeX fractions (\\frac{a}{b} -> (a)/(b))
    - Normalizes LaTeX sqrt (\\sqrt{x} -> (x)^(1/2))
    - Collapses unnecessary spaces around operators
    """
    if not expr_str:
        return ""

    s = expr_str.strip()

    # Strip enclosing LaTeX delimiters $...$ or $$...$$
    s = re.sub(r"^\${1,2}(.*?)\${1,2}$", r"\1", s).strip()

    # Convert unicode superscripts first
    s = normalize_superscripts(s)

    # LaTeX fractions: \frac{num}{den} -> ((num)/(den))
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)

    # LaTeX sqrt: \sqrt{x} -> (x)^(1/2)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"((\1)^(1/2))", s)

    # LaTeX symbols
    s = s.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
    s = s.replace(r"\pm", "+-").replace(r"\leq", "<=").replace(r"\geq", ">=")

    # Unicode operators
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")

    # Exponents: replace ** with ^ for unified string view
    s = s.replace("**", "^")

    # Remove extra whitespace around operators
    s = re.sub(r"\s*([\+\-\*\/\^=<>])\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def parse_to_sympy(expr_str: str):
    """
    Parse a mathematical expression or equation into a SymPy object safely.
    Returns sympy.Expr, sympy.Eq, or None if parsing fails.
    """
    if not HAS_SYMPY or not expr_str:
        return None

    norm = normalize_math_expression(expr_str)

    # Handle equations 'LHS = RHS'
    if "=" in norm and "!=" not in norm and "<=" not in norm and ">=" not in norm:
        parts = norm.split("=")
        if len(parts) == 2:
            lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
            try:
                lhs = parse_expr(lhs_str, transformations=SYMPY_TRANSFORMATIONS, evaluate=False)
                rhs = parse_expr(rhs_str, transformations=SYMPY_TRANSFORMATIONS, evaluate=False)
                return sympy.Eq(lhs, rhs)
            except Exception:
                pass

    try:
        return parse_expr(norm, transformations=SYMPY_TRANSFORMATIONS, evaluate=False)
    except Exception:
        return None


def are_math_expressions_equivalent(expr1_str: str, expr2_str: str) -> bool:
    """
    Determine if two math expressions or equations are mathematically equivalent.
    Uses SymPy symbolic simplification:
    - Expressions: simplify(e1 - e2) == 0
    - Equations: simplify((LHS1 - RHS1) - (LHS2 - RHS2)) == 0 or proportional
    Falls back to canonical string comparison if SymPy fails.
    """
    if not expr1_str or not expr2_str:
        return False

    norm1 = normalize_math_expression(expr1_str)
    norm2 = normalize_math_expression(expr2_str)

    if norm1 == norm2:
        return True

    if not HAS_SYMPY:
        # Fallback to normalized alphanumeric strip
        clean1 = re.sub(r"\s+", "", norm1)
        clean2 = re.sub(r"\s+", "", norm2)
        return clean1 == clean2

    sym1 = parse_to_sympy(expr1_str)
    sym2 = parse_to_sympy(expr2_str)

    if sym1 is None or sym2 is None:
        # Fallback to normalized string comparison
        return norm1.replace(" ", "") == norm2.replace(" ", "")

    try:
        # Case 1: Both are equations
        if isinstance(sym1, sympy.Eq) and isinstance(sym2, sympy.Eq):
            diff1 = sympy.simplify(sym1.lhs - sym1.rhs)
            diff2 = sympy.simplify(sym2.lhs - sym2.rhs)
            if sympy.simplify(diff1 - diff2) == 0 or sympy.simplify(diff1 + diff2) == 0:
                return True
            # Check proportionality (e.g. 2x = 4 equivalent to x = 2)
            try:
                ratio = sympy.simplify(diff1 / diff2)
                if ratio.is_number and ratio != 0:
                    return True
            except Exception:
                pass
            return False

        # Case 2: Both are expressions
        if not isinstance(sym1, sympy.Eq) and not isinstance(sym2, sympy.Eq):
            diff = sympy.simplify(sym1 - sym2)
            return diff == 0

    except Exception:
        pass

    return norm1.replace(" ", "") == norm2.replace(" ", "")
