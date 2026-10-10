
from datasets import load_dataset

DATASET_NAME = "artem9k/ai-text-detection-pile"

print("Loading external dataset information...")

try:
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    print("\nDataset columns:")
    print(dataset.column_names)

    print("\nFirst 3 records:")
    for i, record in enumerate(dataset.take(3), start=1):
        print(f"\nRecord {i}")
        for key, value in record.items():
            value = str(value)
            print(f"{key}: {value[:250]}")

except Exception as error:
    print("\nDataset loading failed:")
    print(type(error).__name__, str(error))
