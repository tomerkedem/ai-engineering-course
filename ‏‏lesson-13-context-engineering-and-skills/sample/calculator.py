"""A small calculator module used to test-drive the unittest-writer skill."""


def add(a, b):
    """Return the sum of two numbers."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("add() requires numeric arguments")
    return a + b


def divide(a, b):
    """Return a divided by b."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("divide() requires numeric arguments")
    if b == 0:
        raise ZeroDivisionError("cannot divide by zero")
    return a / b


def factorial(n):
    """Return n! for a non-negative integer n."""
    if not isinstance(n, int):
        raise TypeError("factorial() requires an integer")
    if n < 0:
        raise ValueError("factorial() is not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def is_palindrome(text):
    """Return True if text reads the same forwards and backwards (ignoring case)."""
    if not isinstance(text, str):
        raise TypeError("is_palindrome() requires a string")
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
