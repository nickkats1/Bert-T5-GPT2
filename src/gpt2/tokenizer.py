from transformers import AutoTokenizer, PreTrainedTokenizerBase

from gpt2.config import ClmModelArguments


def load_tokenizer(model_args: ClmModelArguments) -> PreTrainedTokenizerBase:
    """Load the tokenizer and register the configured pad token."""
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_special_tokens({"pad_token": model_args.pad_token})

    return tokenizer
