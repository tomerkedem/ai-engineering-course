---
name: unittest-writer
description: Write unit tests for a specific Python file. Use when the user asks to add, generate, or verify unit tests for Python code.
---

# unittest-writer

Write high-quality unit tests for a single Python file using Python's built-in
`unittest` package. Follow the rules and flow below exactly. Do not skip the
approval step, and always run the tests at the end.

## Rules

1. **At least 3 tests per function.** Every public function/method in the target
   file must have a minimum of 3 test cases.
2. **Happy-flow and negative tests.** For each function include at least one
   happy-path test (valid input, expected result) and at least one negative test
   (invalid input, error handling, edge cases such as empty/None/boundary values).
3. **Clear, explainable test names.** Each test method name must describe the
   scenario and expected outcome, e.g.
   `test_add_returns_sum_for_two_positive_numbers`,
   `test_divide_raises_zerodivisionerror_when_denominator_is_zero`.
   Never use vague names like `test_1` or `test_add`.
4. **Use the `unittest` package.** Tests subclass `unittest.TestCase`, use
   `self.assert*` methods and `with self.assertRaises(...)` for errors. Do not use
   pytest-style bare `assert` or third-party test frameworks.

## Flow

Follow these steps in order.

### 1. Examine the Python files and suggest a target
- List the Python source files in the project (ignore `test_*.py`, `*_test.py`,
  virtual environments, and generated code).
- Pick the file that most benefits from tests (core logic, most functions, least
  existing coverage) and **suggest it to the user**, briefly explaining why.
- If the user already named a file, use that one.

### 2. Check whether tests already exist
- Look for an existing test file for the target, e.g. `test_<module>.py`,
  `<module>_test.py`, or files under a `tests/` directory that import the target.

### 3a. If tests exist -> verify coverage
- Parse the target file's functions and the existing test file.
- For each function, count how many tests exercise it.
- Report which functions have **fewer than 3 tests**, and whether each has both a
  happy-flow and a negative test.
- Propose the specific additional tests needed to reach the minimum. Then go to
  step 4 (approval) before writing them.

### 3b. If tests don't exist -> create from scratch
- Identify every function/method in the target file.
- Design 3+ tests per function (happy flow + negative), following the rules.

### 4. Show the test plan for approval (MANDATORY)
- Before writing any code, present the tests you intend to add as a **table** with
  the columns: `#`, `Function`, `Test name`, `Type` (happy/negative), and
  `What it checks` (one line). Group rows by function. Example:

  | # | Function | Test name | Type | What it checks |
  |---|----------|-----------|------|----------------|
  | 1 | add | test_add_returns_sum_for_two_positive_numbers | happy | `add(2, 3) == 5` |
  | 2 | add | test_add_returns_correct_value_for_negative_and_positive | happy | `add(-5, 2) == -3` |
  | 3 | add | test_add_raises_typeerror_when_arg_is_a_string | negative | `TypeError` on string arg |

- After the table, state the totals (tests per function) to make the "≥3 per
  function" rule easy to audit.
- **Wait for the user's approval** (or requested changes) before implementing.
  Do not write the test file until the user approves the table.

### 5. Implement the tests
- Confirm the target file's current path right before writing (it may have moved).
- Place the test file **in the same directory as the target module** so the import
  resolves, and import the module by name (e.g. `from calculator import add`).
- Create or update the test file using `unittest`, name the test class after the
  unit under test (e.g. `class TestCalculator(unittest.TestCase)`), and include the
  `if __name__ == "__main__": unittest.main()` guard.

### 6. Run the tests and verify they pass
- Run **from the directory containing the test and target files** so imports
  resolve: `cd <that dir> && python -m unittest <test_module> -v`
  (or `python -m unittest discover -v`). If the source lives elsewhere, adjust the
  working directory or `sys.path` accordingly.
- State the exact command and directory used.
- Show the output. If any test fails, determine whether the bug is in the test or
  the source, fix the test if it is wrong, and re-run until all tests pass.
- Confirm to the user that all tests pass, reporting the count.

## Test skeleton

```python
import unittest

from mymodule import add


class TestAdd(unittest.TestCase):
    def test_add_returns_sum_for_two_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_returns_negative_when_result_is_below_zero(self):
        self.assertEqual(add(-5, 2), -3)

    def test_add_raises_typeerror_when_arg_is_a_string(self):
        with self.assertRaises(TypeError):
            add(2, "3")


if __name__ == "__main__":
    unittest.main()
```
