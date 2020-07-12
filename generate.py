import warnings
warnings.filterwarnings('ignore')

import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import pandas as pd

torch.manual_seed(1234)

df = pd.read_csv('./data/humor_text_only.csv')
txt = df['text'].tolist()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id = tokenizer.eos_token_id)

# Sentence: I think my iphone is broken i keep pressing the home button but i am still at work
input_ids = tokenizer.encode(txt[758], return_tensors = 'pt')
sample_outputs = model.generate(
    input_ids,
    do_sample = True,
    max_length = 50,
    top_k = 50,
    top_p = 0.95,
    num_return_sequences = 2
)

# for t in txt:
#     input_ids = tokenizer.encode(t, return_tensors = 'pt')
#     sample_outputs = model.generate(
#         input_ids,
#         do_sample = True,
#         max_length = 50,
#         top_k = 50,
#         top_p = 0.95,
#         num_return_sequences = 2
#     )

print("Output:\n" + 7 * '=')

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens = True)))

# Output:
# =======
# 0: I think my iphone is broken i keep pressing the home button but i am still at work and need a new one.


# I haven't changed my phone on all my work devices. I've been told it needs a new phone and
# 1: I think my iphone is broken i keep pressing the home button but i am still at work in the morning on my iPhone. i just found out after reading this that its working OK.


# Also, i have another broken phone with 3