from dataclasses import dataclass


@dataclass(frozen=True)
class T5CFG:
    """Config for t5 model and summarization training"""

    model_name: str = "t5-base"

    output_dir: str = "t5-base-reuters"

    data_path: str = "data/reuters_headlines.csv"

    source_prefix: str = "summarize: "

    max_source_length: int = 128

    max_target_length: int = 32

    num_train_epochs: int = 3

    per_device_train_batch_size: int = 8

    per_device_eval_batch_size: int = 4

    gradient_accumulation_steps: int = 3

    learning_rate: float = 5e-5

    weight_decay: float = 0.01

    optim: str = "adamw_torch"

    logging_strategy: str = "steps"

    eval_strategy: str = "epoch"

    save_strategy: str = "epoch"

    fp16: bool = False

    save_total_limit: int = 2

    load_best_model_at_end: bool = True

    metric_for_best_model: str = "rouge1"

    greater_is_better: bool = True

    predict_with_generate: bool = True

    generation_num_beams: int = 1

    logging_steps: int = 20

    seed: int = 42

    report_to: str = "none"

    remove_unused_columns: bool = False
