from transformers import pipeline

# DistilBERT finetuned on SST-2
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # cpu
)

# initial testing
examples = [
    "This movie was an absolute masterpiece.",
    "Terrible acting, nonsensical plot.",
    "It was fine, nothing special.",
]

results = classifier(examples)
for text, result in zip(examples, results):
    print(f"[{result['label']} | {result['score']:.2f}] {text}")
