import json
import traceback

CATEGORY_MULTIPLIERS = {"food": 0.9, "lux": 1.2, "book": 0.95}
DISCOUNT_MULTIPLIERS = {"SAVE10": 0.9, "SAVE20": 0.8, "HALF": 0.5}
VIP_MULTIPLIER = 0.95
BULK_THRESHOLD = 1000
BULK_REBATE = 50


def calculate_line_total(item):
    """Return the price for a single order line, adjusted for its category.

    Args:
        item: A dict with numeric ``q`` (quantity) and ``p`` (unit price) and a
            string ``cat`` (category).

    Returns:
        The line total as a float.

    Raises:
        ValueError: If required keys are missing or values are not numeric.
    """
    if not isinstance(item, dict):
        raise ValueError("item must be a dict")
    for key in ("q", "p", "cat"):
        if key not in item:
            raise ValueError(f"item is missing required key: {key}")
    quantity = item["q"]
    unit_price = item["p"]
    if not isinstance(quantity, (int, float)) or not isinstance(unit_price, (int, float)):
        raise ValueError("item quantity and price must be numeric")
    line_total = quantity * unit_price
    return line_total * CATEGORY_MULTIPLIERS.get(item["cat"], 1.0)


def apply_discount_code(amount, discount_code):
    """Return ``amount`` after applying the given discount code (if recognized).

    Args:
        amount: The pre-discount total as a number.
        discount_code: A discount code string, or None for no discount.

    Returns:
        The discounted amount as a float.

    Raises:
        ValueError: If ``amount`` is not numeric.
    """
    if not isinstance(amount, (int, float)):
        raise ValueError("amount must be numeric")
    return amount * DISCOUNT_MULTIPLIERS.get(discount_code, 1.0)


def apply_vip_and_bulk(amount, user):
    """Return ``amount`` after applying the VIP discount and bulk rebate.

    Args:
        amount: The current total as a number.
        user: A dict that may contain a boolean ``vip`` flag.

    Returns:
        The adjusted amount as a float.

    Raises:
        ValueError: If ``amount`` is not numeric or ``user`` is not a dict.
    """
    if not isinstance(amount, (int, float)):
        raise ValueError("amount must be numeric")
    if not isinstance(user, dict):
        raise ValueError("user must be a dict")
    if user.get("vip"):
        amount *= VIP_MULTIPLIER
    if amount > BULK_THRESHOLD:
        amount -= BULK_REBATE
    return amount


def calculate_order_total(items, user, tax_rate, discount_code):
    """Calculate the final order total for a user, including tax and discounts.

    Args:
        items: A list of order-line dicts (see :func:`calculate_line_total`).
        user: A dict with at least a ``name`` and optional ``vip`` flag.
        tax_rate: The tax rate as a non-negative float (e.g. 0.17).
        discount_code: A discount code string, or None.

    Returns:
        The final total as a float.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    if not isinstance(items, list):
        raise ValueError("items must be a list")
    if not isinstance(user, dict) or "name" not in user:
        raise ValueError("user must be a dict with a 'name'")
    if not isinstance(tax_rate, (int, float)) or tax_rate < 0:
        raise ValueError("tax_rate must be a non-negative number")

    total = sum(calculate_line_total(item) for item in items)
    total = apply_discount_code(total, discount_code)
    total += total * tax_rate
    total = apply_vip_and_bulk(total, user)
    print(f"total for {user['name']} is {total}")
    return total


def load_orders(path):
    """Load and return JSON order data from ``path``.

    Args:
        path: A non-empty path string to a JSON file.

    Returns:
        The parsed data, or None if the file could not be read.

    Raises:
        ValueError: If ``path`` is not a non-empty string.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    try:
        with open(path) as file_handle:
            return json.load(file_handle)
    except Exception:
        traceback.print_exc()
        return None


def save_orders(path, orders):
    """Write ``orders`` to ``path`` as JSON.

    Args:
        path: A non-empty path string to write to.
        orders: A JSON-serializable object.

    Returns:
        True if the write succeeded, False otherwise.

    Raises:
        ValueError: If ``path`` is not a non-empty string.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    try:
        with open(path, "w") as file_handle:
            json.dump(orders, file_handle)
        return True
    except Exception:
        traceback.print_exc()
        return False
