"""
Create SQLite database from dataset_design schema and sample data.
Idempotent: overwrites store.db on each run.
Run from project root with: python database/init_db.py
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "store.db"

# ---------------------------------------------------------------------------
# Schema (from dataset_design/03_sql_schema_definition.md)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    address_line1 TEXT,
    address_line2 TEXT,
    city TEXT NOT NULL,
    state_or_province TEXT,
    postal_code TEXT NOT NULL,
    country TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock_quantity INTEGER NOT NULL,
    is_digital INTEGER NOT NULL,
    is_clearance INTEGER NOT NULL,
    sku TEXT
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    status TEXT NOT NULL,
    total_amount REAL NOT NULL,
    shipping_cost REAL,
    shipping_address_snapshot TEXT,
    delivery_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    line_total REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE shipments (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    tracking_number TEXT NOT NULL,
    carrier TEXT,
    shipped_at TEXT NOT NULL,
    delivered_at TEXT,
    status TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

CREATE TABLE returns (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    requested_at TEXT NOT NULL,
    reason TEXT,
    status TEXT NOT NULL,
    refunded_at TEXT,
    refund_amount REAL,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

CREATE TABLE payments (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    method TEXT NOT NULL,
    status TEXT NOT NULL,
    transaction_id TEXT,
    created_at TEXT NOT NULL,
    failure_reason TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

CREATE TABLE support_tickets (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_id INTEGER,
    subject TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

-- Indexes (from schema doc)
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_delivery_date ON orders(delivery_date);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
CREATE INDEX idx_shipments_order_id ON shipments(order_id);
CREATE INDEX idx_shipments_tracking_number ON shipments(tracking_number);
CREATE INDEX idx_returns_order_id ON returns(order_id);
CREATE INDEX idx_payments_order_id ON payments(order_id);
CREATE INDEX idx_payments_status ON payments(status);
"""

# ---------------------------------------------------------------------------
# Sample data (from dataset_design/04_generated_dataset_samples.md)
# ---------------------------------------------------------------------------

# 20 customers: id, name, email, phone, address_line1, address_line2, city, state_or_province, postal_code, country, signup_date, is_active
# UK row (id 15) has state "—" → None; plausible address_line1, phone added where not in doc
CUSTOMERS = [
    (1, "Emma Johnson", "emma.j@email.com", "555-0101", "100 Congress Ave", None, "Austin", "TX", "78701", "US", "2023-01-15", 1),
    (2, "Marcus Chen", "m.chen@email.com", "555-0102", "400 Broad St", None, "Seattle", "WA", "98101", "US", "2023-03-22", 1),
    (3, "Sofia Rodriguez", "sofia.r@email.com", "555-0103", "200 Biscayne Blvd", None, "Miami", "FL", "33101", "US", "2023-05-10", 1),
    (4, "James Wilson", "j.wilson@email.com", "555-0104", "1 Beacon St", None, "Boston", "MA", "02101", "US", "2023-06-18", 1),
    (5, "Olivia Brown", "olivia.b@email.com", "555-0105", "1600 Broadway", None, "Denver", "CO", "80201", "US", "2023-07-05", 1),
    (6, "Liam Taylor", "l.taylor@email.com", "555-0106", "233 S Wacker Dr", None, "Chicago", "IL", "60601", "US", "2023-08-12", 1),
    (7, "Ava Martinez", "ava.m@email.com", "555-0107", "200 W Washington St", None, "Phoenix", "AZ", "85001", "US", "2023-09-20", 1),
    (8, "Noah Anderson", "noah.a@email.com", "555-0108", "250 Williams St", None, "Atlanta", "GA", "30301", "US", "2023-10-01", 1),
    (9, "Isabella Thomas", "isabella.t@email.com", "555-0109", "1507 Main St", None, "Dallas", "TX", "75201", "US", "2023-10-15", 1),
    (10, "Lucas Jackson", "lucas.j@email.com", "555-0110", "1 Market St", None, "San Francisco", "CA", "94102", "US", "2023-11-01", 1),
    (11, "Mia White", "mia.w@email.com", "555-0111", "1000 SW Broadway", None, "Portland", "OR", "97201", "US", "2023-11-20", 1),
    (12, "Ethan Harris", "ethan.h@email.com", "555-0112", "50 S 6th St", None, "Minneapolis", "MN", "55401", "US", "2023-12-05", 1),
    (13, "Charlotte Clark", "charlotte.c@email.com", "555-0113", "1 King St W", None, "Toronto", "ON", "M5V 1A1", "Canada", "2024-01-10", 1),
    (14, "Alexander Lewis", "a.lewis@email.com", "555-0114", "200 Granville St", None, "Vancouver", "BC", "V6B 1A1", "Canada", "2024-02-14", 1),
    (15, "Amelia Walker", "amelia.w@email.com", "555-0115", "10 Downing St", None, "London", None, "SW1A 1AA", "UK", "2024-02-28", 1),
    (16, "Henry Hall", "henry.h@email.com", "555-0116", "350 5th Ave", None, "New York", "NY", "10001", "US", "2024-03-01", 1),
    (17, "Harper Young", "harper.y@email.com", "555-0117", "1 World Way", None, "Los Angeles", "CA", "90001", "US", "2024-03-05", 1),
    (18, "Benjamin King", "benjamin.k@email.com", "555-0118", "1600 Smith St", None, "Houston", "TX", "77001", "US", "2024-03-08", 1),
    (19, "Evelyn Wright", "evelyn.w@email.com", "555-0119", "1 Liberty Pl", None, "Philadelphia", "PA", "19101", "US", "2024-03-10", 1),
    (20, "Sebastian Lopez", "sebastian.l@email.com", "555-0120", "600 B St", None, "San Diego", "CA", "92101", "US", "2024-03-12", 1),
]

# 20 products: id, name, description, category, price, stock_quantity, is_digital, is_clearance, sku
PRODUCTS = [
    (1, "Wireless Bluetooth Headphones", None, "Electronics", 79.99, 45, 0, 0, None),
    (2, "Cotton T-Shirt (Unisex)", None, "Clothing", 24.99, 120, 0, 0, None),
    (3, "Stainless Steel Kitchen Set", None, "Home & Kitchen", 89.99, 30, 0, 0, None),
    (4, "Python Programming E-Book", None, "Books", 29.99, 0, 1, 0, None),
    (5, "Kids Building Blocks Set", None, "Toys", 34.99, 55, 0, 0, None),
    (6, "Running Shoes (Men's)", None, "Sports", 89.99, 40, 0, 0, None),
    (7, "Coffee Maker 12-Cup", None, "Home & Kitchen", 59.99, 25, 0, 0, None),
    (8, "Classic Novel (Paperback)", None, "Books", 14.99, 200, 0, 0, None),
    (9, "Yoga Mat", None, "Sports", 29.99, 80, 0, 0, None),
    (10, "USB-C Hub 7-in-1", None, "Electronics", 49.99, 60, 0, 0, None),
    (11, "Winter Jacket (Women's)", None, "Clothing", 119.99, 35, 0, 0, None),
    (12, "Digital Music Album Download", None, "Electronics", 9.99, 0, 1, 0, None),
    (13, "Desk Lamp LED", None, "Home & Kitchen", 39.99, 70, 0, 0, None),
    (14, "Board Game Family", None, "Toys", 44.99, 42, 0, 0, None),
    (15, "Water Bottle 32oz", None, "Sports", 19.99, 150, 0, 0, None),
    (16, "Last-Season Speaker (Refurb)", None, "Electronics", 45.00, 15, 0, 1, None),
    (17, "Cookbook (Hardcover)", None, "Books", 24.99, 90, 0, 0, None),
    (18, "Kids Sneakers", None, "Clothing", 39.99, 48, 0, 0, None),
    (19, "Online Course: Photography", None, "Electronics", 79.99, 0, 1, 0, None),
    (20, "Throw Pillow Set (2pk)", None, "Home & Kitchen", 32.00, 28, 0, 1, None),
]

# 20 orders: id, customer_id, order_date, status, total_amount, shipping_cost, shipping_address_snapshot, delivery_date
ORDERS = [
    (1001, 1, "2025-03-14 10:22:00", "Pending", 104.98, None, None, None),
    (1002, 1, "2025-03-10 09:15:00", "Shipped", 54.99, None, None, None),
    (1003, 2, "2025-01-28 14:30:00", "Delivered", 129.98, None, None, "2025-02-05"),
    (1004, 2, "2025-03-15 11:00:00", "Pending", 89.99, None, None, None),
    (1005, 3, "2025-03-01 16:45:00", "Delivered", 44.99, None, None, "2025-03-11"),
    (1006, 3, "2025-02-20 08:00:00", "Delivered", 164.97, None, None, "2025-02-24"),
    (1007, 4, "2025-03-12 13:20:00", "Shipped", 119.99, None, None, None),
    (1008, 5, "2025-02-10 10:00:00", "Delivered", 59.99, None, None, "2025-02-18"),
    (1009, 6, "2025-03-16 09:05:00", "Pending", 149.98, None, None, None),
    (1010, 7, "2025-03-08 17:30:00", "Delivered", 79.99, None, None, "2025-03-11"),
    (1011, 8, "2025-02-01 12:00:00", "Delivered", 95.00, None, None, "2025-02-10"),
    (1012, 9, "2025-03-11 14:00:00", "Delivered", 29.99, None, None, "2025-03-11"),
    (1013, 10, "2025-03-13 08:30:00", "Pending", 199.97, None, None, None),
    (1014, 11, "2025-01-15 09:00:00", "Delivered", 34.99, None, None, "2025-01-22"),
    (1015, 12, "2025-03-15 16:00:00", "Pending", 64.98, None, None, None),
    (1016, 13, "2024-12-20 11:00:00", "Delivered", 89.99, None, None, "2024-12-28"),
    (1017, 14, "2025-03-10 10:00:00", "Shipped", 119.99, None, None, None),
    (1018, 15, "2025-02-25 15:00:00", "Delivered", 44.99, None, None, "2025-03-02"),
    (1019, 16, "2025-03-14 12:00:00", "Pending", 39.99, None, None, None),
    (1020, 17, "2025-03-09 17:00:00", "Delivered", 159.98, None, None, "2025-03-11"),
]

# 16 order_items: id, order_id, product_id, quantity, unit_price, line_total
ORDER_ITEMS = [
    (1, 1001, 1, 1, 79.99, 79.99),
    (2, 1001, 2, 1, 24.99, 24.99),
    (3, 1002, 9, 1, 29.99, 29.99),
    (4, 1002, 15, 1, 19.99, 19.99),
    (5, 1003, 6, 1, 89.99, 89.99),
    (6, 1003, 10, 1, 39.99, 39.99),
    (7, 1005, 14, 1, 44.99, 44.99),
    (8, 1006, 3, 1, 89.99, 89.99),
    (9, 1006, 7, 1, 59.99, 59.99),
    (10, 1006, 15, 1, 19.99, 19.99),
    (11, 1010, 1, 1, 79.99, 79.99),
    (12, 1013, 3, 1, 89.99, 89.99),
    (13, 1013, 11, 1, 119.99, 119.99),
    (14, 1004, 6, 1, 89.99, 89.99),
    (15, 1009, 1, 1, 79.99, 79.99),
    (16, 1009, 6, 1, 89.99, 89.99),
]

# 5 shipments: id, order_id, tracking_number, carrier, shipped_at, delivered_at, status
SHIPMENTS = [
    (1, 1002, "1Z999AA10123456784", "UPS", "2025-03-10 14:00:00", None, "In Transit"),
    (2, 1005, "9400111899223344556677", "USPS", "2025-03-02 09:00:00", "2025-03-11 14:30:00", "Delivered"),
    (3, 1007, "FD1234567890", "FedEx", "2025-03-13 08:00:00", None, "In Transit"),
    (4, 1017, "1Z999BB20234567895", "UPS", "2025-03-11 10:00:00", None, "In Transit"),
    (5, 1006, "9400111899223344556688", "USPS", "2025-02-21 11:00:00", "2025-02-24 16:00:00", "Delivered"),
]

# 2 returns: id, order_id, requested_at, reason, status, refunded_at, refund_amount
RETURNS = [
    (1, 1008, "2025-02-25 10:00:00", "Changed mind", "Refunded", "2025-03-02 14:00:00", 59.99),
    (2, 1014, "2025-02-01 09:00:00", "Damaged", "Refunded", "2025-02-08 11:00:00", 34.99),
]

# 5 payments: id, order_id, amount, method, status, transaction_id, created_at, failure_reason
PAYMENTS = [
    (1, 1001, 104.98, "Credit Card", "Completed", None, "2025-03-14 10:22:00", None),
    (2, 1004, 89.99, "Credit Card", "Completed", None, "2025-03-15 11:00:00", None),
    (3, 1013, 199.97, "Credit Card", "Failed", None, "2025-03-13 08:30:00", "Card declined"),
    (4, 1015, 64.98, "PayPal", "Completed", None, "2025-03-15 16:00:00", None),
    (5, 1003, 129.98, "Credit Card", "Completed", None, "2025-01-28 14:30:00", None),
]


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL)
    cur = conn.cursor()

    cur.executemany(
        "INSERT INTO customers (id, name, email, phone, address_line1, address_line2, city, state_or_province, postal_code, country, signup_date, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        CUSTOMERS,
    )
    cur.executemany(
        "INSERT INTO products (id, name, description, category, price, stock_quantity, is_digital, is_clearance, sku) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        PRODUCTS,
    )
    cur.executemany(
        "INSERT INTO orders (id, customer_id, order_date, status, total_amount, shipping_cost, shipping_address_snapshot, delivery_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ORDERS,
    )
    cur.executemany(
        "INSERT INTO order_items (id, order_id, product_id, quantity, unit_price, line_total) VALUES (?, ?, ?, ?, ?, ?)",
        ORDER_ITEMS,
    )
    cur.executemany(
        "INSERT INTO shipments (id, order_id, tracking_number, carrier, shipped_at, delivered_at, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        SHIPMENTS,
    )
    cur.executemany(
        "INSERT INTO returns (id, order_id, requested_at, reason, status, refunded_at, refund_amount) VALUES (?, ?, ?, ?, ?, ?, ?)",
        RETURNS,
    )
    cur.executemany(
        "INSERT INTO payments (id, order_id, amount, method, status, transaction_id, created_at, failure_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        PAYMENTS,
    )
    # support_tickets: no rows

    conn.commit()
    conn.close()
    print(f"Created {DB_PATH}")


if __name__ == "__main__":
    main()
