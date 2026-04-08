### Sentiment Analysis
Fine-tuning DistilBERT on the IMDB dataset for binary sentiment classification.
- HuggingFace `pipeline` API for zero-shot inference
- Tokenization with `AutoTokenizer` and dynamic padding
- Fine-tuning with `Trainer` + `TrainingArguments`
- Evaluation with `evaluate` library (accuracy, F1)
