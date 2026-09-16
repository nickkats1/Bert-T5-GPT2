class TestLoadTokenizer:
    def test_has_a_pad_token(self, gpt2_tokenizer):
        assert gpt2_tokenizer.pad_token is not None

    def test_pad_token_is_not_the_eos_token(self, gpt2_tokenizer):
        assert gpt2_tokenizer.pad_token_id != gpt2_tokenizer.eos_token_id

    def test_pad_token_is_the_configured_one(self, gpt2_model_args, gpt2_tokenizer):
        assert gpt2_tokenizer.pad_token == gpt2_model_args.pad_token

    def test_can_pad_a_batch(self, gpt2_tokenizer):
        encoded = gpt2_tokenizer(["short", "a much longer headline here"], padding=True)

        assert len(encoded["input_ids"][0]) == len(encoded["input_ids"][1])
