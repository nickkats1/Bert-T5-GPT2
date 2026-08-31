from dataclasses import dataclass


@dataclass(frozen=True)
class Gpt2CFG:
    """Config for gpt2 model and headline generation training"""

    model_name: str = "gpt2"

    output_dir: str = "gpt2-headlines"

    data_path: str = "data/reuters_headlines.csv"

    pad_token: str = "<|pad|>"

    max_length: int = 64

    num_train_epochs: int = 2

    per_device_train_batch_size: int = 8

    per_device_eval_batch_size: int = 4

    gradient_accumulation_steps: int = 2

    learning_rate: float = 5e-5

    weight_decay: float = 0.01

    optim: str = "adamw_torch"

    logging_strategy: str = "steps"

    eval_strategy: str = "epoch"

    save_strategy: str = "epoch"

    fp16: bool = False

    save_total_limit: int = 2

    load_best_model_at_end: bool = True

    metric_for_best_model: str = "loss"

    greater_is_better: bool = False

    max_new_tokens: int = 32

    num_return_sequences: int = 1

    logging_steps: int = 20

    seed: int = 42

    report_to: str = "none"

    remove_unused_columns: bool = False
