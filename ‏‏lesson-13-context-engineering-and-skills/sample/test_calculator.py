"""Unit tests for calculator.py (written with the unittest-writer skill)."""

import unittest

from calculator import add, divide, factorial, is_palindrome


class TestAdd(unittest.TestCase):
    def test_add_returns_sum_for_two_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_returns_correct_value_for_negative_and_positive(self):
        self.assertEqual(add(-5, 2), -3)

    def test_add_raises_typeerror_when_arg_is_a_string(self):
        with self.assertRaises(TypeError):
            add(2, "3")


class TestDivide(unittest.TestCase):
    def test_divide_returns_quotient_for_two_positive_numbers(self):
        self.assertEqual(divide(10, 2), 5)

    def test_divide_returns_float_for_non_even_division(self):
        self.assertEqual(divide(7, 2), 3.5)

    def test_divide_raises_zerodivisionerror_when_denominator_is_zero(self):
        with self.assertRaises(ZeroDivisionError):
            divide(1, 0)

    def test_divide_raises_typeerror_when_arg_is_not_numeric(self):
        with self.assertRaises(TypeError):
            divide("10", 2)


class TestFactorial(unittest.TestCase):
    def test_factorial_returns_one_for_zero(self):
        self.assertEqual(factorial(0), 1)

    def test_factorial_returns_product_for_positive_integer(self):
        self.assertEqual(factorial(5), 120)

    def test_factorial_raises_valueerror_for_negative_number(self):
        with self.assertRaises(ValueError):
            factorial(-3)

    def test_factorial_raises_typeerror_for_non_integer(self):
        with self.assertRaises(TypeError):
            factorial(3.5)


class TestIsPalindrome(unittest.TestCase):
    def test_is_palindrome_returns_true_for_simple_palindrome(self):
        self.assertTrue(is_palindrome("level"))

    def test_is_palindrome_ignores_case_and_spaces(self):
        self.assertTrue(is_palindrome("Never odd or even"))

    def test_is_palindrome_returns_false_for_non_palindrome(self):
        self.assertFalse(is_palindrome("hello"))

    def test_is_palindrome_raises_typeerror_for_non_string(self):
        with self.assertRaises(TypeError):
            is_palindrome(12321)


if __name__ == "__main__":
    unittest.main()
