# Part 3 — SQL Schema Definition (Conceptual)

Conceptual relational schema for the online store. No SQL code—only table names, field names, types (conceptual), and relationships.

---

## Core Tables

### customers

| Field | Conceptual type | Description |
|-------|-----------------|-------------|
| id | unique identifier | Primary key. |
| name | string | Full name. |
| email | string | Unique; used for login and notifications. |
| phone | string (optional) | Phone number. |
| address_line1 | string | Street address. |
| address_line2 | string (optional) | Apt, suite, etc. |
| city | string | City. |
| state_or_province | string | State / province / region. |
| postal_code | string | ZIP / postal code. |
| country | string | ISO or full name (e.g., "US", "Canada"). |
| signup_date | date or datetime | When the account was created. |
| is_active | boolean (optional) | Whether the account is active. |

---

### products

| Field | Conceptual type | Description |
|-------|-----------------|-------------|
| id | unique identifier | Primary key. |
| name | string | Product name. |
| description | string (optional) | Short description. |
| category | string | e.g., Electronics, Clothing, Home & Kitchen, Books, Toys, Sports. |
| price | decimal | Unit price. |
| stock_quantity | integer | Current stock. |
| is_digital | boolean | True for e-books, downloads, etc. (non-returnable per policy). |
| is_clearance | boolean | True for final-sale / clearance items. |
| sku | string (optional) | Stock keeping unit. |

---

### orders

| Field | Conceptual type | Description |
|-------|-----------------|-------------|
| id | unique identifier | Primary key; displayed as order number. |
| customer_id | foreign key → customers.id | Who placed the order. |
| order_date | datetime | When the order was placed. |
| status | enum / string | One of: Pending, Processing, Shipped, Delivered, Cancelled, Returned. |
| total_amount | decimal | Total after tax/shipping (or subtotal; define consistently). |
| shipping_cost | decimal (optional) | Shipping fee. |
| shipping_address_snapshot | string or JSON (optional) | Address at time of order. |
| delivery_date | date (optional) | Actual or estimated delivery date; used for return eligibility (e.g. 30 days from delivery). |

**Note:** `status` and `delivery_date` are critical for policy alignment: cancellation only when Pending; returns within 30 days of delivery.

---

## Junction / Detail Tables

### order_items

Links orders to products with quantity and price at time of order.

| Field | Conceptual type | Description |
|-------|-----------------|-------------|
| id | unique identifier | Primary key. |
| order_id | foreign key → orders.id | Order. |
| product_id | foreign key → products.id | Product. |
| quantity | integer | Quantity ordered. |
| unit_price | decimal | Price per unit at time of order. |
| line_total | decimal | quantity × unit_price (or stored). |

---

### shipments

One order can have multiple shipments (e.g. multiple packages or partial fulfillment).

| Field | Conceptual type | Description |
|-------|-----------------|-------------|
| id | unique identifier | Primary key. |
| order_id | foreign key → orders.id | Order. |
| tracking_number | string | Carrier tracking number. |
| carrier | string (optional) | e.g., USPS, FedEx, UPS. |
| shipped_at | datetime | When the shipment was dispatched. |
| delivered_at | datetime (optional) | When the carrier marked it delivered. |
| status | string (optional) | e.g., In Transit, Delivered, Exception. |

**Use:** Lost-package and delivery-delay scenarios; "where is my order?" answers.

---

### returns

Tracks return requests and outcomes for return/refund policy reasoning.

| Field | Conceptual type | Description |
|-------|-----------------|-------------|
| id | unique identifier | Primary key. |
| order_id | foreign key → orders.id | Order being returned. |
| requested_at | datetime | When the customer requested the return. |
| reason | string (optional) | e.g., Damaged, Wrong item, Changed mind. |
| status | string | e.g., Pending, Received, Refunded, Rejected. |
| refunded_at | datetime (optional) | When refund was processed. |
| refund_amount | decimal (optional) | Amount refunded. |

---

### payments

Links payments to orders for payment-failure and refund scenarios.

| Field | Conceptual type | Description |
|-------|-----------------|-------------|
| id | unique identifier | Primary key. |
| order_id | foreign key → orders.id | Order. |
| amount | decimal | Amount charged or refunded. |
| method | string | e.g., Credit Card, PayPal, Gift Card. |
| status | string | e.g., Completed, Failed, Refunded, Pending. |
| transaction_id | string (optional) | External gateway reference. |
| created_at | datetime | When the payment was attempted or completed. |
| failure_reason | string (optional) | Stored if status = Failed (e.g., "Insufficient funds", "Card declined"). |

---

### support_tickets (optional)

For scenarios like "I already opened a ticket" or future ticket-based workflows.

| Field | Conceptual type | Description |
|-------|-----------------|-------------|
| id | unique identifier | Primary key. |
| customer_id | foreign key → customers.id | Customer. |
| order_id | foreign key → orders.id (optional) | Related order. |
| subject | string | Short subject. |
| status | string | e.g., Open, In Progress, Resolved, Closed. |
| created_at | datetime | When the ticket was opened. |
| updated_at | datetime (optional) | Last update. |

---

## Entity Relationship Summary

- **customers** 1 — N **orders**
- **orders** N — N **products** via **order_items**
- **orders** 1 — N **shipments**
- **orders** 1 — N **returns** (typically 0 or 1 per order for simplicity)
- **orders** 1 — N **payments**
- **customers** 1 — N **support_tickets**; **support_tickets** optionally link to **orders**

---

## Indexing Hints (for later implementation)

- `orders.customer_id`, `orders.status`, `orders.delivery_date`
- `order_items.order_id`, `order_items.product_id`
- `shipments.order_id`, `shipments.tracking_number`
- `returns.order_id`
- `payments.order_id`, `payments.status`

This schema supports answering: return eligibility (order + delivery_date + products.is_digital/is_clearance), cancellation (orders.status), shipment status (shipments), payment issues (payments), and international shipping (customers.country or shipping address).
