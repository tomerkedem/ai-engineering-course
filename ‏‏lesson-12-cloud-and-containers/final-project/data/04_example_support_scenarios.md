# Part 5 — Example Customer Support Scenarios

Fifteen to twenty example user questions, with **which policies (RAG)** and **which database data** the agent would need to answer each. The agent combines: (1) retrieval from the policy knowledge base, (2) queries to the SQL database (e.g. after resolving customer identity and order context).

---

## Scenario 1

**User question:** "Can I return my last order?"

**Policies needed (RAG):**
- **Return Policy** — 30-day window from delivery, condition (unused, original packaging), proof of purchase.
- **Refund Policy** — how refunds are processed and timing.

**Database needed:**
- Identify **last order** for the customer (e.g. `orders` sorted by `order_date` DESC, limit 1).
- Get **delivery_date** for that order. If `delivery_date` is null (e.g. not yet delivered), agent says returns apply after delivery.
- Compare `delivery_date` to today: if within 30 days → eligible; if past 30 days → not eligible.
- Optionally join **order_items** and **products** to check if any item is `is_digital` or `is_clearance`; if so, cite exclusions from Return Policy.

**Example:** Customer 3’s last order 1006 delivered 2025-02-24 (20 days ago) → eligible. Customer 2’s last order 1003 delivered 2025-02-05 → not eligible (past 30 days).

---

## Scenario 2

**User question:** "My order arrived damaged. What should I do?"

**Policies needed (RAG):**
- **Damaged Products** — contact within 7 days, order number, description of damage; replacement or full refund; do not discard packaging.

**Database needed:**
- Confirm the order exists and belongs to the customer (e.g. `orders` by `customer_id` and order id/number).
- Check **order status** (e.g. Delivered) and **delivery_date** to confirm it has arrived and is within the 7-day window for reporting damage.
- Optionally check **returns** for that order to see if a return is already in progress.

**Example:** Order 1010 delivered 2025-03-11 → within 7 days; agent gives steps and asks for order number and damage description.

---

## Scenario 3

**User question:** "Why can't I cancel my order?"

**Policies needed (RAG):**
- **Order Cancellation** — orders can only be cancelled if status is **Pending** (not yet shipped); once shipped, must use return process.

**Database needed:**
- Get the specific order (e.g. by order number or “my order”) and its **status**.
- If `status` = Pending → agent explains how to cancel (or that they can).
- If `status` = Shipped, Processing, or Delivered → agent explains that cancellation is not possible and suggests return policy if they want to send it back.

**Example:** Order 1002 (Shipped) or 1007 (Shipped) → “Your order has already shipped, so we can’t cancel it. You can return it after delivery within 30 days if it meets our return policy.”

---

## Scenario 4

**User question:** "Do you ship to Canada?"

**Policies needed (RAG):**
- **International Shipping** — we ship to Canada, UK, Australia, EU; costs and delivery times at checkout; customer responsible for customs/duties.

**Database needed:**
- None strictly required for a generic “do you ship to Canada?” (policy only).
- If the user is logged in, **customers.country** can personalize (“Yes, we ship to Canada. Your account is in Canada, so you can place an order and see shipping options at checkout.”).

**Example:** Customer 13 or 14 (Canada) → agent confirms we ship to Canada and can mention their address is in Canada.

---

## Scenario 5

**User question:** "Where is my shipment?"

**Policies needed (RAG):**
- **Shipping Policy** or **Delivery Delays** — typical timeframes; when to contact if delayed.
- **Lost Packages** — when we consider a package lost (e.g. 7 business days domestic with no update).

**Database needed:**
- Get customer’s **orders** with status Shipped or Delivered.
- Join **shipments** on `order_id` to get **tracking_number**, **carrier**, **shipped_at**, **delivered_at**, **status**.
- Return tracking link and status; if “In Transit” and long past expected delivery, mention Lost Packages policy.

**Example:** Order 1002 → shipment 1 (UPS, In Transit, tracking 1Z999…). Order 1005 → shipment 2 (USPS, Delivered 2025-03-11).

---

## Scenario 6

**User question:** "Why was my payment rejected?"

**Policies needed (RAG):**
- **Payment Failures** — common reasons (insufficient funds, wrong card details, bank decline, billing address); what to do (verify details, try another method, contact bank); we don’t see exact decline code.

**Database needed:**
- **payments** for the customer’s order(s): filter `status` = Failed.
- **failure_reason** if stored (e.g. “Card declined”, “Insufficient funds”) to give a more specific hint without revealing internal details.
- **orders** to confirm which order was not placed or is on hold.

**Example:** Order 1013 has payment Failed, failure_reason “Card declined” → agent explains common causes and suggests verifying card and billing address or trying another payment method.

---

## Scenario 7

**User question:** "Can I return an item I bought 2 months ago?"

**Policies needed (RAG):**
- **Return Policy** — returns within 30 days of delivery only; no returns after that.

**Database needed:**
- Find the order (and line item if needed) and its **delivery_date**.
- If delivery was more than 30 days ago → not eligible; agent states the 30-day rule.

**Example:** Order 1003 delivered 2025-02-05 (e.g. ~40 days ago) → “Returns are allowed within 30 days of delivery. Your order was delivered on [date], so it’s outside that window and we can’t process a return.”

---

## Scenario 8

**User question:** "I want to exchange my shirt for a different size."

**Policies needed (RAG):**
- **Exchange Policy** — same 30-day-from-delivery and condition rules as returns; process (request exchange, pay/refund difference); not for clearance/final sale.
- **Return Policy** — eligibility (30 days, unused, original packaging).

**Database needed:**
- Order and **delivery_date** to confirm within 30 days.
- **order_items** + **products** to confirm the product is not **is_clearance** (exchanges not available for final sale).
- Confirm no existing **returns** that would conflict.

**Example:** Order 1005 (delivered 2025-03-11) with a Clothing item → eligible; agent explains how to request an exchange (account or support).

---

## Scenario 9

**User question:** "Do you ship internationally?"

**Policies needed (RAG):**
- **International Shipping** — countries (Canada, UK, Australia, EU), delivery times, customs/duties, restrictions.

**Database needed:**
- Optional: **customers.country** if user is logged in, to tailor the answer (e.g. “We ship to the UK, including your country.”).

---

## Scenario 10

**User question:** "My package says delivered but I never received it."

**Policies needed (RAG):**
- **Lost Packages** — delivered but not received: we work with carrier; may require checking with neighbors; after investigation we may offer one-time replacement or refund.

**Database needed:**
- **shipments** for the order: **delivered_at**, **carrier**, **tracking_number**.
- **orders** to confirm order and delivery date.
- Agent uses this to acknowledge “we see it was marked delivered on [date]” and then follows policy (investigation steps, then resolution).

---

## Scenario 11

**User question:** "How long until I get my refund?"

**Policies needed (RAG):**
- **Refund Policy** — 5–7 business days after we receive the return; then 3–10 days for bank to post.

**Database needed:**
- **returns** for the order: **status** (e.g. Received, Refunded), **refunded_at**.
- If status = Refunded and **refunded_at** is recent → “Your refund was processed on [date]; allow 3–10 business days for your bank.”
- If status = Pending or Received → “We’re processing your return; refunds are issued within 5–7 business days after we receive the item.”

**Example:** Return 1 (order 1008) Refunded 2025-03-02 → agent gives timing from that date.

---

## Scenario 12

**User question:** "Can I change my shipping address? I just placed the order."

**Policies needed (RAG):**
- **Order Modifications** — address can only be changed if order is still **Pending** (not shipped); if already shipped, we can’t change destination.

**Database needed:**
- **orders**: **status** and **order_date** (or **shipments.shipped_at** if any).
- If status = Pending and no shipment → “We can update the address. Please contact us with the new address and order number.”
- If Shipped → “Your order has already shipped; we can’t change the address. You may need to contact the carrier or refuse delivery and place a new order.”

---

## Scenario 13

**User question:** "My order is late. When will it arrive?"

**Policies needed (RAG):**
- **Delivery Delays** — check tracking first; we work with carrier; possible shipping refund for guaranteed delivery not met.
- **Shipping Policy** — typical delivery windows.

**Database needed:**
- **shipments**: **shipped_at**, **delivered_at**, **status**, **carrier**, **tracking_number**.
- **orders**: **order_date**, **delivery_date** (if estimated).
- Agent summarizes current status and suggests checking tracking; if past promised date, mentions Delivery Delays and possible shipping refund.

---

## Scenario 14

**User question:** "I bought an e-book by mistake. Can I get a refund?"

**Policies needed (RAG):**
- **Digital Products** — once accessed/downloaded, sales are final; no refund for change of mind; only defective or not-as-described.
- **Return Policy** — digital products non-returnable once accessed.

**Database needed:**
- **order_items** + **products**: confirm the item is **is_digital** (e.g. product_id 4, 12, 19).
- If digital and already delivered/accessed → agent explains that digital products are non-refundable after access, except for defective or not-as-described (and how to report those).

---

## Scenario 15

**User question:** "The item I want to return was on clearance. Can I still return it?"

**Policies needed (RAG):**
- **Clearance and Final Sale** — clearance/final sale items are non-returnable (except defective or wrong item).
- **Return Policy** — exclusions (clearance/final sale).

**Database needed:**
- **order_items** + **products**: **is_clearance** (or similar) for the item.
- If clearance → agent states that clearance items are final sale and cannot be returned, unless defective or wrong item received.

---

## Scenario 16

**User question:** "Can I use a promo code on an existing order?"

**Policies needed (RAG):**
- **Promotions and Discounts** — one code per order; applied at checkout; we don’t apply codes to already-placed orders.
- **Order Modifications** — what can be changed on pending orders.

**Database needed:**
- **orders**: **status**. If Pending, agent can say we can’t add a promo to an existing order but they could cancel and reorder with the code (if cancellation is allowed). If not Pending, same message without cancel option.

---

## Scenario 17

**User question:** "I can’t log in to my account. I forgot my password."

**Policies needed (RAG):**
- **Account Issues** — “Forgot password” on login page; email with link; link expires 24 hours; check spam; account lock after failed attempts.

**Database needed:**
- Optional: **customers** to verify email exists (agent never confirms/denies existence for security; generic instructions are enough).

---

## Scenario 18

**User question:** "Is my order eligible for return? I got it about three weeks ago."

**Policies needed (RAG):**
- **Return Policy** — 30 days from delivery; condition and packaging.

**Database needed:**
- **orders** for the customer: **delivery_date**.
- Compute days since delivery. “About three weeks” = ~21 days → likely within 30 days; agent can say “If your order was delivered within the last 30 days and the items are unused and in original packaging, you can return them,” and suggest they confirm exact delivery date in their order details.

---

## Scenario 19

**User question:** "I was charged but my order says payment failed. What happened?"

**Policies needed (RAG):**
- **Payment Failures** — sometimes a hold appears before decline; we don’t charge for failed orders; hold usually drops in a few days; contact bank if it doesn’t.

**Database needed:**
- **payments**: **order_id**, **status** (Failed), **amount**, **created_at**.
- **orders**: confirm order status (e.g. not placed or Pending payment).
- Agent explains that failed payments may show a temporary hold and that we didn’t complete the charge; suggest checking bank and retrying with a different method if needed.

---

## Scenario 20

**User question:** "Do you offer warranty on the headphones I bought?"

**Policies needed (RAG):**
- **Warranty** — most products 1 year from purchase against defects; how to claim (contact support, order number, possibly return); what’s not covered (misuse, wear and tear).

**Database needed:**
- **order_items** + **products**: confirm product (e.g. headphones) and that it’s a physical product.
- **orders**: **order_date** to indicate warranty start date.
- Agent states standard 1-year warranty and how to make a claim (support, order number).

---

## Summary table

| # | Question type | Main policy doc(s) | Main DB table(s) |
|---|----------------|--------------------|-------------------|
| 1 | Return last order | Return, Refund | orders, order_items, products |
| 2 | Damaged order | Damaged Products | orders, returns |
| 3 | Can’t cancel | Order Cancellation | orders |
| 4 | Ship to Canada? | International Shipping | customers (optional) |
| 5 | Where is shipment? | Shipping, Lost Packages | orders, shipments |
| 6 | Payment rejected | Payment Failures | payments, orders |
| 7 | Return 2 months ago | Return Policy | orders |
| 8 | Exchange for size | Exchange, Return | orders, order_items, products |
| 9 | Ship internationally? | International Shipping | customers (optional) |
| 10 | Delivered but not received | Lost Packages | shipments, orders |
| 11 | Refund timing | Refund Policy | returns |
| 12 | Change address | Order Modifications | orders, shipments |
| 13 | Order late | Delivery Delays, Shipping | shipments, orders |
| 14 | E-book refund | Digital Products | order_items, products |
| 15 | Return clearance item | Clearance, Return | order_items, products |
| 16 | Promo on existing order | Promotions, Order Modifications | orders |
| 17 | Forgot password | Account Issues | (optional: customers) |
| 18 | Eligible for return? | Return Policy | orders |
| 19 | Charged but payment failed | Payment Failures | payments, orders |
| 20 | Warranty | Warranty | order_items, products, orders |

These scenarios cover returns, refunds, exchanges, cancellation, shipping, international, damaged/lost packages, payment issues, digital/clearance, account, and warranty—using both the RAG knowledge base and the relational database as specified.
