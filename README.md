# PayGuard: Fraud Detection and Complaint Triage

## Project Overview

This project addresses two problems for PayGuard, a fictional digital payments company:

- **Part A:** Detect fraudulent card transactions using machine learning.
- **Part B:** Classify and summarize customer complaints using an LLM.
- **Part C:** Provide recommendations for production deployment.

## Part A: Fraud Detection

**Dataset:** Credit Card Fraud Detection Dataset (Kaggle).

The dataset contains 284,807 transactions, including 492 fraudulent transactions (0.173%). No missing values were found.

I used a chronological data split of 70% training, 15% validation, and 15% testing to evaluate the model on later transactions.

I compared two models:

- **Dummy Classifier:** Baseline model that predicts every transaction as genuine.
- **Logistic Regression:** Stronger model using StandardScaler and balanced class weights to address class imbalance.

### Model Comparison

| Metric                       | Dummy Classifier | Logistic Regression |
| ---------------------------- | ---------------- | ------------------- |
| Validation Accuracy          | 99.87%           | 97.82%              |
| Validation Precision         | 0.00%            | 5.30%               |
| Validation Recall            | 0.00%            | 92.86%              |
| Validation F1-score          | 0.00%            | 10.03%              |
| Validation Average Precision | 0.13%            | 83.94%              |

The Dummy Classifier achieved high accuracy because almost all transactions were genuine. However, it failed to detect any fraud.

Logistic Regression detected substantially more fraudulent transactions and achieved much higher average precision. This demonstrates why accuracy alone is misleading for imbalanced fraud datasets.

### Threshold Selection

I evaluated Logistic Regression using different thresholds on the validation set.

| Threshold | Precision | Recall | F1-score | False Positives | False Negatives |
| --------- | --------- | ------ | -------- | --------------- | --------------- |
| 0.50      | 5.30%     | 92.86% | 10.03%   | 929             | 4               |
| 0.70      | 11.00%    | 91.10% | 19.70%   | 411             | 5               |
| 0.80      | 17.20%    | 91.10% | 28.90%   | 246             | 5               |
| 0.90      | 27.20%    | 89.30% | 41.70%   | 134             | 6               |
| 0.95      | 35.50%    | 87.50% | 50.50%   | 89              | 7               |

**Selected Model:** Logistic Regression with a threshold of **0.95**.

The selected threshold reduced false positives from 929 to 89 while maintaining 87.50% recall on validation data.

False positives increase unnecessary customer friction and manual review workload, while false negatives allow fraudulent transactions to go undetected.

I selected 0.95 as a reasonable starting threshold for a human-reviewed fraud alert system. The threshold should be reassessed using actual fraud losses and review costs before production deployment.

### Final Test Results

The selected model was evaluated on the held-out test set.

| Metric            | Result |
| ----------------- | ------ |
| Accuracy          | 99.84% |
| Precision         | 42.11% |
| Recall            | 76.92% |
| F1-score          | 54.42% |
| Average Precision | 70.69% |
| True Positives    | 40     |
| True Negatives    | 42,615 |
| False Positives   | 55     |
| False Negatives   | 12     |

The model detected 40 out of 52 fraudulent transactions. However, 12 fraud cases were missed and 55 genuine transactions were incorrectly flagged.

These results support further testing but do not justify automatic transaction blocking.

### Real-Time Inference Considerations

For real-time fraud detection, the model must receive transaction features in the same format and order used during training.

The preprocessing pipeline should remain consistent between training and inference.

The dataset contains anonymized PCA features (V1–V28), which would need to be available and calculated consistently in a production environment. The dataset's `Time` feature also requires reconsideration because it represents elapsed time within the original dataset.

Only features available at the moment of a transaction should be used to avoid data leakage.

A production system would also require low-latency predictions, monitoring, and periodic evaluation for changes in fraud patterns.

## Part B: Complaint Triage

**Dataset:** CCDB_Export_15_January_2026_through_February_2026.csv

I cleaned the dataset and randomly selected 400 complaints.

**LLM Used:** GPT-OSS-20B through the Groq API.

The system processes complaints and returns structured JSON with the following fields:

- `category`
- `urgency`
- `summary`
- `requires_human`
- `confidence_note`

Python validates the JSON output and applies additional human-review checks for sensitive complaints.

### Evaluation and Limitations

I prepared 400 complaints for evaluation. However, Groq free-tier API rate limits prevented the complete batch from being processed.

Out of 10 attempted complaints, 8 were successfully classified and 2 were handled through error fallbacks.

These results represent a small pilot and do not establish the system's overall routing accuracy.

### API Cost Estimation

The API cost was estimated using the following formula:

**Cost per Complaint = [(Input Tokens × Input Price) + (Output Tokens × Output Price)] / 1,000,000**

Assumed token usage and pricing:

- Input tokens: 800
- Output tokens: 200
- Input price: $0.075 per 1 million tokens
- Output price: $0.30 per 1 million tokens

**Cost per Complaint:**

(800 × 0.075 + 200 × 0.30) / 1,000,000 = **$0.00012**

**Monthly Cost for 50,000 Complaints:**

0.00012 × 50,000 = **$6.00**

This is an estimated API cost based on assumed token usage and pricing. Actual usage was not measured. Retries, infrastructure, and human-review costs are not included.

## Part C: Production Recommendation

Both solutions show potential but need further testing before full deployment.

For fraud detection, I recommend using Logistic Regression with human review of flagged transactions instead of automatic blocking. The selected threshold should be monitored and adjusted according to operational costs and missed fraud cases.

For complaint triage, I recommend human-supervised routing, especially for fraud-related, legal, and uncertain complaints.

Both systems should be monitored for performance, errors, and reliability.

A more detailed production recommendation is provided in `memo.md`.

## Project Setup

**Requirements:** Python, pandas, numpy, scikit-learn, joblib, groq, and python-dotenv.

For complaint triage, add the `GROQ_API_KEY` to a local `.env` file. Do not include the API key in the submission.

### Run Part A

```bash
cd fraud_detection
python train.py
python predict.py
```

### Run Part B

```bash
cd complain_triage
python prepare_data.py
python batch_triage.py
```

API rate limits should be considered before running larger batches.

## AI Usage

ChatGPT was used for code generation, debugging, implementation guidance, and documentation assistance.

The problem understanding, solution decisions, evaluation interpretation, and final responsibility were mine.

Further details are provided in `AI_USAGE.md`.

## Conclusion

This project demonstrates how machine learning and LLMs can support fraud detection and customer complaint management.

Logistic Regression performed significantly better than the Dummy Classifier at identifying fraudulent transactions. The selected model was evaluated on a held-out test set and saved for inference.

Complaint triage was tested on a limited sample due to API restrictions.

Both solutions require further validation and human oversight before production deployment.
