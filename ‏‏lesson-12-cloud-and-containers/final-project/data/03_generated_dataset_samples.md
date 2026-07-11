# Part 4 — Generated Dataset Samples

**Reference date for "today":** 2025-03-16 (used for delivery-day calculations).

Generated to align with policies:
- **Returns:** Orders with delivery_date 5, 20, and 40+ days ago (eligible, eligible, ineligible).
- **Cancellation:** Mix of Pending, Shipped, Delivered so the agent can say "only Pending can be cancelled."
- **International:** Some customers in Canada, UK for "Do you ship to Canada?" etc.
- **Digital/Clearance:** Some products marked digital or clearance for policy exceptions.

---

## 1. Customers (20 rows)

| id | name | email | city | state_or_province | postal_code | country | signup_date |
|----|------|-------|------|-------------------|-------------|---------|-------------|
| 1 | Emma Johnson | emma.j@email.com | Austin | TX | 78701 | US | 2023-01-15 |
| 2 | Marcus Chen | m.chen@email.com | Seattle | WA | 98101 | US | 2023-03-22 |
| 3 | Sofia Rodriguez | sofia.r@email.com | Miami | FL | 33101 | US | 2023-05-10 |
| 4 | James Wilson | j.wilson@email.com | Boston | MA | 02101 | US | 2023-06-18 |
| 5 | Olivia Brown | olivia.b@email.com | Denver | CO | 80201 | US | 2023-07-05 |
| 6 | Liam Taylor | l.taylor@email.com | Chicago | IL | 60601 | US | 2023-08-12 |
| 7 | Ava Martinez | ava.m@email.com | Phoenix | AZ | 85001 | US | 2023-09-20 |
| 8 | Noah Anderson | noah.a@email.com | Atlanta | GA | 30301 | US | 2023-10-01 |
| 9 | Isabella Thomas | isabella.t@email.com | Dallas | TX | 75201 | US | 2023-10-15 |
| 10 | Lucas Jackson | lucas.j@email.com | San Francisco | CA | 94102 | US | 2023-11-01 |
| 11 | Mia White | mia.w@email.com | Portland | OR | 97201 | US | 2023-11-20 |
| 12 | Ethan Harris | ethan.h@email.com | Minneapolis | MN | 55401 | US | 2023-12-05 |
| 13 | Charlotte Clark | charlotte.c@email.com | Toronto | ON | M5V 1A1 | Canada | 2024-01-10 |
| 14 | Alexander Lewis | a.lewis@email.com | Vancouver | BC | V6B 1A1 | Canada | 2024-02-14 |
| 15 | Amelia Walker | amelia.w@email.com | London | — | SW1A 1AA | UK | 2024-02-28 |
| 16 | Henry Hall | henry.h@email.com | New York | NY | 10001 | US | 2024-03-01 |
| 17 | Harper Young | harper.y@email.com | Los Angeles | CA | 90001 | US | 2024-03-05 |
| 18 | Benjamin King | benjamin.k@email.com | Houston | TX | 77001 | US | 2024-03-08 |
| 19 | Evelyn Wright | evelyn.w@email.com | Philadelphia | PA | 19101 | US | 2024-03-10 |
| 20 | Sebastian Lopez | sebastian.l@email.com | San Diego | CA | 92101 | US | 2024-03-12 |

*(address_line1, address_line2, phone can be filled with plausible values; country is set for international shipping scenarios.)*

---

## 2. Products (20 rows)

| id | name | category | price | stock_quantity | is_digital | is_clearance |
|----|------|----------|-------|----------------|------------|--------------|
| 1 | Wireless Bluetooth Headphones | Electronics | 79.99 | 45 | false | false |
| 2 | Cotton T-Shirt (Unisex) | Clothing | 24.99 | 120 | false | false |
| 3 | Stainless Steel Kitchen Set | Home & Kitchen | 89.99 | 30 | false | false |
| 4 | Python Programming E-Book | Books | 29.99 | 0 | true | false |
| 5 | Kids Building Blocks Set | Toys | 34.99 | 55 | false | false |
| 6 | Running Shoes (Men's) | Sports | 89.99 | 40 | false | false |
| 7 | Coffee Maker 12-Cup | Home & Kitchen | 59.99 | 25 | false | false |
| 8 | Classic Novel (Paperback) | Books | 14.99 | 200 | false | false |
| 9 | Yoga Mat | Sports | 29.99 | 80 | false | false |
| 10 | USB-C Hub 7-in-1 | Electronics | 49.99 | 60 | false | false |
| 11 | Winter Jacket (Women's) | Clothing | 119.99 | 35 | false | false |
| 12 | Digital Music Album Download | Electronics | 9.99 | 0 | true | false |
| 13 | Desk Lamp LED | Home & Kitchen | 39.99 | 70 | false | false |
| 14 | Board Game Family | Toys | 44.99 | 42 | false | false |
| 15 | Water Bottle 32oz | Sports | 19.99 | 150 | false | false |
| 16 | Last-Season Speaker (Refurb) | Electronics | 45.00 | 15 | false | true |
| 17 | Cookbook (Hardcover) | Books | 24.99 | 90 | false | false |
| 18 | Kids Sneakers | Clothing | 39.99 | 48 | false | false |
| 19 | Online Course: Photography | Electronics | 79.99 | 0 | true | false |
| 20 | Throw Pillow Set (2pk) | Home & Kitchen | 32.00 | 28 | false | true |

*Categories: Electronics, Clothing, Home & Kitchen, Books, Toys, Sports. Digital and clearance items support policy exceptions (non-returnable / final sale).*

---

## 3. Orders (100 rows) — Summary and policy-aligned subsets

**Status distribution (conceptual):**
- Pending: ~15 orders (cancellable)
- Processing: ~5
- Shipped: ~20 (not cancellable; in transit)
- Delivered: ~55 (return window depends on delivery_date)
- Cancelled: ~3
- Returned: ~2

**Delivery-date alignment for returns (examples):**
- Delivered **5 days ago** (2025-03-11): e.g. orders 12, 34, 56 → **eligible** for return (within 30 days).
- Delivered **20 days ago** (2025-02-24): e.g. orders 7, 22, 41 → **eligible** for return.
- Delivered **40+ days ago** (2025-02-05 or earlier): e.g. orders 3, 18, 29 → **not eligible** (past 30 days).

**Sample orders (subset with key fields):**

| order_id | customer_id | order_date | status | total_amount | delivery_date | Note |
|----------|-------------|------------|--------|--------------|---------------|------|
| 1001 | 1 | 2025-03-14 10:22:00 | Pending | 104.98 | — | Cancellable |
| 1002 | 1 | 2025-03-10 09:15:00 | Shipped | 54.99 | — | In transit |
| 1003 | 2 | 2025-01-28 14:30:00 | Delivered | 129.98 | 2025-02-05 | Past 30 days, no return |
| 1004 | 2 | 2025-03-15 11:00:00 | Pending | 89.99 | — | Cancellable |
| 1005 | 3 | 2025-03-01 16:45:00 | Delivered | 44.99 | 2025-03-11 | Delivered 5 days ago, return OK |
| 1006 | 3 | 2025-02-20 08:00:00 | Delivered | 164.97 | 2025-02-24 | Delivered 20 days ago, return OK |
| 1007 | 4 | 2025-03-12 13:20:00 | Shipped | 119.99 | — | Cannot cancel |
| 1008 | 5 | 2025-02-10 10:00:00 | Delivered | 59.99 | 2025-02-18 | Past 30 days |
| 1009 | 6 | 2025-03-16 09:05:00 | Pending | 149.98 | — | Cancellable |
| 1010 | 7 | 2025-03-08 17:30:00 | Delivered | 79.99 | 2025-03-11 | 5 days ago, return OK |
| … | … | … | … | … | … | … |

*(Full 100 orders would list all ids 1001–1100 with varied customer_id, order_date, status, total_amount, delivery_date; the table above illustrates the pattern.)*

**Expanded sample — 20 full order rows (to show variety):**

| id | customer_id | order_date | status | total_amount | delivery_date |
|----|-------------|------------|--------|--------------|---------------|
| 1001 | 1 | 2025-03-14 10:22:00 | Pending | 104.98 | null |
| 1002 | 1 | 2025-03-10 09:15:00 | Shipped | 54.99 | null |
| 1003 | 2 | 2025-01-28 14:30:00 | Delivered | 129.98 | 2025-02-05 |
| 1004 | 2 | 2025-03-15 11:00:00 | Pending | 89.99 | null |
| 1005 | 3 | 2025-03-01 16:45:00 | Delivered | 44.99 | 2025-03-11 |
| 1006 | 3 | 2025-02-20 08:00:00 | Delivered | 164.97 | 2025-02-24 |
| 1007 | 4 | 2025-03-12 13:20:00 | Shipped | 119.99 | null |
| 1008 | 5 | 2025-02-10 10:00:00 | Delivered | 59.99 | 2025-02-18 |
| 1009 | 6 | 2025-03-16 09:05:00 | Pending | 149.98 | null |
| 1010 | 7 | 2025-03-08 17:30:00 | Delivered | 79.99 | 2025-03-11 |
| 1011 | 8 | 2025-02-01 12:00:00 | Delivered | 95.00 | 2025-02-10 |
| 1012 | 9 | 2025-03-11 14:00:00 | Delivered | 29.99 | 2025-03-11 |
| 1013 | 10 | 2025-03-13 08:30:00 | Pending | 199.97 | null |
| 1014 | 11 | 2025-01-15 09:00:00 | Delivered | 34.99 | 2025-01-22 |
| 1015 | 12 | 2025-03-15 16:00:00 | Pending | 64.98 | null |
| 1016 | 13 | 2024-12-20 11:00:00 | Delivered | 89.99 | 2024-12-28 |
| 1017 | 14 | 2025-03-10 10:00:00 | Shipped | 119.99 | null |
| 1018 | 15 | 2025-02-25 15:00:00 | Delivered | 44.99 | 2025-03-02 |
| 1019 | 16 | 2025-03-14 12:00:00 | Pending | 39.99 | null |
| 1020 | 17 | 2025-03-09 17:00:00 | Delivered | 159.98 | 2025-03-11 |

---

## 4. Order items (sample — multiple products per order)

Each order should have 1–4 line items. Examples:

| id | order_id | product_id | quantity | unit_price | line_total |
|----|----------|------------|----------|------------|------------|
| 1 | 1001 | 1 | 1 | 79.99 | 79.99 |
| 2 | 1001 | 2 | 1 | 24.99 | 24.99 |
| 3 | 1002 | 9 | 1 | 29.99 | 29.99 |
| 4 | 1002 | 15 | 1 | 19.99 | 19.99 |
| 5 | 1003 | 6 | 1 | 89.99 | 89.99 |
| 6 | 1003 | 10 | 1 | 39.99 | 39.99 |
| 7 | 1005 | 14 | 1 | 44.99 | 44.99 |
| 8 | 1006 | 3 | 1 | 89.99 | 89.99 |
| 9 | 1006 | 7 | 1 | 59.99 | 59.99 |
| 10 | 1006 | 15 | 1 | 19.99 | 19.99 |
| 11 | 1010 | 1 | 1 | 79.99 | 79.99 |
| 12 | 1013 | 3 | 1 | 89.99 | 89.99 |
| 13 | 1013 | 11 | 1 | 119.99 | 119.99 |
| 14 | 1004 | 6 | 1 | 89.99 | 89.99 |
| 15 | 1009 | 1 | 1 | 79.99 | 79.99 |
| 16 | 1009 | 6 | 1 | 89.99 | 89.99 |

*Ensure some orders include digital (e.g. product_id 4, 12, 19) or clearance (16, 20) so the agent can apply policy exclusions.*

---

## 5. Shipments (sample)

| id | order_id | tracking_number | carrier | shipped_at | delivered_at | status |
|----|----------|-----------------|---------|------------|--------------|--------|
| 1 | 1002 | 1Z999AA10123456784 | UPS | 2025-03-10 14:00:00 | null | In Transit |
| 2 | 1005 | 9400111899223344556677 | USPS | 2025-03-02 09:00:00 | 2025-03-11 14:30:00 | Delivered |
| 3 | 1007 | FD1234567890 | FedEx | 2025-03-13 08:00:00 | null | In Transit |
| 4 | 1017 | 1Z999BB20234567895 | UPS | 2025-03-11 10:00:00 | null | In Transit |
| 5 | 1006 | 9400111899223344556688 | USPS | 2025-02-21 11:00:00 | 2025-02-24 16:00:00 | Delivered |

---

## 6. Returns (sample)

| id | order_id | requested_at | reason | status | refunded_at | refund_amount |
|----|----------|--------------|--------|--------|-------------|---------------|
| 1 | 1008 | 2025-02-25 10:00:00 | Changed mind | Refunded | 2025-03-02 14:00:00 | 59.99 |
| 2 | 1014 | 2025-02-01 09:00:00 | Damaged | Refunded | 2025-02-08 11:00:00 | 34.99 |

---

## 7. Payments (sample — including one failure)

| id | order_id | amount | method | status | created_at | failure_reason |
|----|----------|--------|--------|--------|------------|----------------|
| 1 | 1001 | 104.98 | Credit Card | Completed | 2025-03-14 10:22:00 | null |
| 2 | 1004 | 89.99 | Credit Card | Completed | 2025-03-15 11:00:00 | null |
| 3 | 1013 | 199.97 | Credit Card | Failed | 2025-03-13 08:30:00 | Card declined |
| 4 | 1015 | 64.98 | PayPal | Completed | 2025-03-15 16:00:00 | null |
| 5 | 1003 | 129.98 | Credit Card | Completed | 2025-01-28 14:30:00 | null |

*(Order 1013 has a failed payment — supports "Why was my payment rejected?" scenario.)*

---

## Policy + database alignment summary

| Policy rule | Data that triggers it |
|-------------|------------------------|
| Returns within 30 days of delivery | Orders 1005, 1006, 1010, 1012: delivery_date within last 30 days. Orders 1003, 1008, 1014, 1016: delivery_date > 30 days ago. |
| Only Pending orders can be cancelled | 1001, 1004, 1009, 1013, 1015, 1019: status = Pending. 1002, 1007, 1017: Shipped (cannot cancel). |
| Digital / clearance non-returnable | Orders containing product_id 4, 12, 19 (digital) or 16, 20 (clearance). |
| International shipping | Customers 13, 14 (Canada), 15 (UK) — agent can say we ship there and use address for eligibility. |
| Payment declined | Order 1013: payment status Failed, failure_reason "Card declined". |
| Lost package / tracking | Shipments with status In Transit and shipped_at sufficiently in the past; or delivered_at null. |
| Damaged / return already processed | Order 1008 or 1014 with returns table entry (reason, status Refunded). |

This gives the agent concrete rows to query and cite when answering the example support scenarios.
