# Golden Set Labeling Methodology & Annotation Guidelines

## 1. Objective
To construct a rigorous, ground-truth evaluation dataset of 200 customer support messages for **AmazonHelp** (primary deep-dive brand) and 50 messages each for **AppleSupport** and **Uber_Support** (secondary brands) to benchmark intent classification accuracy, retrieval quality, grounded reply faithfulness, and auto-handle vs. escalation decisions.

---

## 2. Intent Taxonomy Definitions (AmazonHelp)

| Intent ID | Intent Name | Description | Keywords / Indicators | Auto-Handle Eligible? |
|---|---|---|---|---|
| `order_tracking_delivery` | Order Tracking & Delivery | Where is my order (WISMO), late packages, delivery carrier delays, tracking link requests, locker pickups. | track, where, package, delivery, order, late, carrier, locker, arrived | ✅ Yes |
| `refund_return_cancellation` | Returns, Refunds & Cancellation | Requesting refunds, returning defective/unwanted items, canceling orders before shipment, return drop-offs. | refund, return, cancel, money back, damaged, broken, return label | ✅ Yes |
| `prime_subscription_billing` | Prime Membership & Billing | Unexpected credit card charges, Prime subscription renews, trial cancellations, payment method failures. | prime, charged, fee, subscription, annual, renew, billing, membership | ✅ Yes |
| `account_security_access` | Account Access & Security | Locked accounts, password resets, 2FA OTP codes, hacked accounts, suspicious activity alerts. | locked, password, otp, 2fa, hacked, security, unauthorized, access | ❌ No (Human Security) |
| `digital_services_devices` | Kindle, Fire TV & Digital Content | Kindle e-reader troubleshooting, Fire TV stick errors, Alexa/Echo setups, Prime Video streaming glitches. | kindle, fire tv, alexa, echo, prime video, stream, app, device | ✅ Yes |
| `seller_product_inquiry` | Third-Party Sellers & Product Info | Marketplace seller questions, warranties, product authenticity, stock availability, specifications. | seller, marketplace, stock, warranty, authentic, counterfeit, price match | ✅ Yes |
| `general_feedback_complaint` | Grievances & Executive Escalation | Extreme frustration, threats of legal action, supervisor demands, general service complaints. | worst, terrible, lawyer, sue, supervisor, manager, unacceptable, scam | ❌ No (Human Escalation) |

---

## 3. Escalation Decision Protocol

A message is labeled as `should_escalate = True` if **ANY** of the following conditions are met:
1. **Security / Privacy Risk**: The query belongs to `account_security_access` (e.g. account takeover, 2FA bypass).
2. **Legal / Regulatory Risk**: The customer mentions "lawyer", "sue", "police", "fraud", "consumer protection", or "BBB".
3. **Severe Dissatisfaction**: Customer exhibits extreme anger or explicit request to speak with a supervisor/manager (VADER sentiment compound score $< -0.45$).
4. **Multi-Turn Frustration**: The customer indicates this is their 3rd+ attempt to resolve the issue without success.

Otherwise, if the intent is auto-handle eligible and standard troubleshooting/tracking can resolve it, the message is labeled as `should_escalate = False`.

---

## 4. Annotation Workflow (LLM-Assisted Pre-labeling + Human Verification)

```
[Customer Inbound Tweet]
          │
          ▼
[LLM Pre-Labeler (Llama-3.1-8B-Instant via Instructor)]
   - Predicted Intent
   - Confidence Score
   - Step-by-Step Reasoning
          │
          ▼
[Audit Logging -> data/golden/{brand}_prelabels.csv]
          │
          ▼
[Deterministic Rule & Sentiment Verification Gate]
   - Security Policy Cross-Check
   - Sentiment Thresholding
          │
          ▼
[Human Verification & Correction]
   - Verify intent edge cases
   - Audit ambiguous queries
          │
          ▼
[Final Golden Dataset -> data/golden/{brand}_golden.csv]
```
