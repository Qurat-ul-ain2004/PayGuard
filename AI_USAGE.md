# AI Usage Disclosure

## AI Tools Used

### 1. ChatGPT (OpenAI)

ChatGPT was used as a development assistant for:

**Part A: Fraud Detection**

- Generating and modifying Python code for data preprocessing and model training.
- Implementing Gaussian Naive Bayes and Logistic Regression.
- Setting up StandardScaler pipelines and chronological data splitting.
- Debugging errors and assisting with model evaluation.
- Implementing threshold comparisons, confusion matrices, and model saving.
- Assisting with inference code.

**Part B: Complaint Triage**

- Assisting with dataset preparation and sampling.
- Developing LLM prompts for complaint classification and summarization.
- Implementing structured JSON output validation and fallback handling.
- Debugging API errors and handling rate limits.
- Assisting with evaluation and API cost estimation.

**Part C: Production Recommendation**

- Assisting with structuring and drafting the production design memo.
- Explaining deployment considerations, monitoring, human review, and operational risks.

**Documentation**

- Assisting with the README, setup instructions, technical explanations, and documentation revisions.

### 2. Groq API (GPT-OSS-20B)

The Groq API was used in Part B to process customer complaint narratives.

The language model generated structured outputs containing complaint categories, urgency, summaries, and human-review indicators.

## Estimated AI-Generated Code

**Approximate share of code generated or substantially modified with ChatGPT assistance: [XX%].**

This estimate covers Python implementation across Parts A and B.&#x20;

## Code Ownership and Verification

I executed the code, inspected model evaluation results, tested inference, and reviewed the implementation decisions.

I compared Gaussian Naive Bayes and Logistic Regression using validation metrics and selected Logistic Regression based on its stronger performance.

I evaluated different classification thresholds and selected 0.95 as a starting point for a human-reviewed fraud alert workflow.

I also tested the complaint triage prototype and documented the limitations caused by API rate limits.

I take responsibility for understanding, explaining, modifying, and maintaining the submitted code.

## Summary

AI assistance was used for code generation, debugging, technical explanations, and documentation. The final implementation decisions, reported results, and responsibility for the submission remain with the author.
