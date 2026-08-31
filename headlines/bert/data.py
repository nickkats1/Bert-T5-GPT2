"""Loading and splitting of the Guardian headline CSV for BERT fine-tuning."""

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from headlines.bert.config import BertCFG


def load_csv(file_path: str) -> Dataset:
    """Load CSV file from file path and returns dataset.

    Args:
        file_path: where the file is located.

    Returns:
        dataset: CSV file converted as dataset instance.
    """
    return Dataset.from_csv(str(file_path))


def split_csv(dataset: Dataset) -> tuple[Dataset, Dataset, Dataset]:
    """Split loaded dataset into train/test/val splits.

    Args:
        dataset: Dataset wrapper for CSV data to be trained.

    Returns:
        tuple consisting of three different datasets for each split.
    """
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    test_val = splits["test"].train_test_split(test_size=0.5, seed=42)
    train = splits["train"]
    test = test_val["train"]
    val = test_val["test"]

    return train, test, val


def build_label_maps(dataset: Dataset) -> tuple[dict[object, int], dict[int, object]]:
    """Build the label/id lookups the model needs to report real class names.

    Args:
        dataset: Dataset carrying a "labels" column.

    Returns:
        tuple of label-to-id and id-to-label mappings.
    """
    labels = sorted(set(dataset["labels"]))

    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = dict(enumerate(labels))

    return label_to_id, id_to_label


def tokenize(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    label_to_id: dict[object, int],
) -> Dataset:
    """Tokenize the headline column and encode labels as ids.

    Args:
        dataset: split to tokenize.
        tokenizer: tokenizer matching the model being fine-tuned.
        label_to_id: mapping produced by build_label_maps.

    Returns:
        dataset with input_ids, attention_mask and encoded labels.
    """

    def preprocess(batch):
        encoded = tokenizer(
            batch["Headlines"],
            truncation=True,
            max_length=BertCFG.max_length,
            padding="max_length",
            return_token_type_ids=False,
        )
        encoded["labels"] = [label_to_id[label] for label in batch["labels"]]
        return encoded

    return dataset.map(preprocess, batched=True, remove_columns=["Headlines"])
