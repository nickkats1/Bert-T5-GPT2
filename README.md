# Headlines: Transformers

This code base trains three popular pre-trained transformers: Bert(encoder-only), T5(seq-to-seq) and gpt2(text-generation) on a dataset consisting of article headlines. I am acting like this is 2021 in terms of relevance. Before the LLM decoder-only part of the transformer really blew up.

## Overview

The main reason for this repo, is to show the history of the transformer and the parts of it that were trained before many things came out in the last four years. Before massive scaling, instruct gpt, RLHF, gpt2 was not very good at all. Other models were much more popular. Bert was heavily used, T5 was groundbreaking and GPT2 just kinda generated nonsense with no structure. These models are important and should not be forgotten.


### Tasks

- **Text Classification**: Bert, Encoder-only part of transformer for text-classification
- **Text Summarization**: T5, encoder-decoder part of the transformer for summarization tasks.
- **Text Generation**: GPT2: Decoder-Only, autoregressive part of the transformer for text-generation tasks.


### Bert (Encoder-Only)

Every token attends to every other token in both directions, so the model builds one representation
of the whole sentence. That is the right shape for classification: a head reads a single label off
that representation. It is not a generator — it has no notion of continuing text. This repo uses
`bert-base-uncased`.


### GPT-2 (decoder-only)

Each token attends only to the tokens before it, and the model is trained to predict the next one.
That is why it generates text, and why its score is perplexity rather than accuracy or ROUGE —
there is no single right answer to compare against, only how surprised the model is by real text.
Since this was before `instruct gpt` and `RLHF`, the model kinda generated incoherent nonsense.


### T5 (encoder-decoder)

The encoder reads the input, and a separate decoder writes the output while attending back to the
encoder through cross-attention. Input and output are both text and do not have to be the same
length, which is what summarization needs: a long description in, a short headline out.



## License

[MIT](/home/nick/github-projects/bert-t5-gpt2/LICENSE)

