# Headlines: Transformers

This code base trains three popular pre-trained transformers: Bert(encoder-only), T5(seq-to-seq) and gpt2(text-generation) on a dataset consisting of article headlines.

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

**Example From Notebook**

```py
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

phrases = [
    "My phantom sit on sixes no 20s in my denim. Your cutlass knocking because it is a lemon.",
    "I like them Georgia peaches but you look more like a lemon.",
    "I'm pimping wearing linen, that's just how I am chilling. I'm smoking grits and selling chickens. Corvette painted lemon.",
    "I got lemonade and lemon tint.",
    "Half a pound of lemon kush, call that pack the lemon drop.",
    "Just stash one lemon, homie, I can supply damn near 20 blocks.",
]

inputs = tokenizer(phrases, padding=True, return_tensors="pt")

with torch.inference_mode():
    hidden_states = model(**inputs).last_hidden_state

lemon_token_id = tokenizer.convert_tokens_to_ids("lemon")
lemon_embeddings = []

for phrase_index, token_ids in enumerate(inputs["input_ids"]):
    lemon_positions = (token_ids == lemon_token_id).nonzero(as_tuple=True)[0]

    for lemon_position in lemon_positions:
        embedding = hidden_states[phrase_index, lemon_position]
        lemon_embeddings.append(embedding)

        print(f"Phrase {phrase_index + 1}")
        print("Token ID:", token_ids[lemon_position].item())
        print("Embedding preview:", embedding[:8].tolist())
        print()

embeddings = torch.stack(lemon_embeddings)
```

### GPT-2 (decoder-only)

Each token attends only to the tokens before it, and the model is trained to predict the next one.
That is why it generates text, and why its score is perplexity rather than accuracy or ROUGE —
there is no single right answer to compare against, only how surprised the model is by real text.
Since this was before `instruct gpt` and `RLHF`, the model kinda generated incoherent nonsense.

**example using Gucci Mane "Lemonade" Lyrics.**

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>",
    "bos_token": "<|startoftext|>",
})

tokenizer.padding_size = "left"

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

phrases = [
    "My phantom sit on sixes no 20s in my denim. Your cutlass knocking because it is a lemon.",
    "I like them Georgia peaches but you look more like a lemon.",
    "I'm pimping wearing linen, that's just how I am chilling. I'm smoking grits and selling chickens. Corvette painted lemon.",
    "I got lemonade and lemon tint.",
    "Half a pound of lemon kush, call that pack the lemon drop.",
    "Just stash one lemon, homie, I can supply damn near 20 blocks.",
]

# tokenize phrases
inputs = tokenizer(phrases, return_tensors="pt", padding=True)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    num_return_sequences=15,
    pad_token_id=tokenizer.pad_token_id,
    temperature=0.7,
)


num_seqs = 15
for p_idx, phrase in enumerate(phrases):
    print(f"==== Phrase {p_idx} ====")
    for s_idx in range(num_seqs):
        out = outputs[p_idx * num_seqs + s_idx]
        print(f"--- sample {s_idx} ---")
        print(tokenizer.decode(out, skip_special_tokens=True))
    print()
```

**output**

```
txt
==== Phrase 0 ====
--- sample 0 ---
My phantom sit on sixes no 20s in my denim. Your cutlass knocking because it is a lemon. that is a red. I am a real red. I am real red. You know, I am not. I am a real red. I am not in a red. I am not in a red. I am not in a red.
--- sample 1 ---
My phantom sit on sixes no 20s in my denim. Your cutlass knocking because it is a lemon. in a room full of people. A lemon is a very good idea, especially to do for a lot of people. I thought it was a lemon and it's a lemon. no no no no no no, it's a lemon.


--- sample 2 ---
My phantom sit on sixes no 20s in my denim. Your cutlass knocking because it is a lemon., I don't care what your fanny or your fanny says. What is the difference between fanny and the fanny? The difference between fanny and a fanny is what you are. You are a fanny because you want to
--- sample 3 ---
My phantom sit on sixes no 20s in my denim. Your cutlass knocking because it is a lemon. at the airport I went to this place with my wife and she went there and she said my son was a little boy. she said he was born at 10 and she said his mother went up to me because the guy was a little girl. he
--- sample 4 ---
My phantom sit on sixes no 20s in my denim. Your cutlass knocking because it is a lemon.

Your car is a car you have only bought from a man for you. You can't walk on the street with your car.

You have taken your dog to a museum. You say it was a really good show, but this
--- sample 5 ---
My phantom sit on sixes no 20s in my denim. Your cutlass knocking because it is a lemon.
```

### T5 (encoder-decoder)

The encoder reads the input, and a separate decoder writes the output while attending back to the
encoder through cross-attention. Input and output are both text and do not have to be the same
length, which is what summarization needs: a long description in, a short headline out.




[License][MIT](/home/nick/github-projects/bert-t5-gpt2/LICENSE)

