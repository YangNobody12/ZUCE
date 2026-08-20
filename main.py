import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map="auto"
)

model.eval()

PROMPT = """
Write a Python function that returns Fibonacci numbers.
"""

def generate():

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

def scale_hook(alpha):

    def hook(module, inp, out):

        if isinstance(out, tuple):
            out = out[0]

        return out * alpha

    return hook

baseline = generate()

print(baseline)

layer = 26

module = model.model.layers[layer]

handle = module.register_forward_hook(
    scale_hook(0.5)
)

text = generate()

handle.remove()

print(text)

from difflib import SequenceMatcher

score = SequenceMatcher(
    None,
    baseline,
    text
).ratio()

print(score)

from difflib import SequenceMatcher

alphas = [1.0,0.8,0.6,0.4,0.2,0.0]

results = []

for layer in range(len(model.model.layers)):

    print("Layer",layer)

    module = model.model.layers[layer]

    for alpha in alphas:

        handle = module.register_forward_hook(
            scale_hook(alpha)
        )

        text = generate()

        handle.remove()

        similarity = SequenceMatcher(
            None,
            baseline,
            text
        ).ratio()

        results.append({

            "layer":layer,
            "alpha":alpha,
            "similarity":similarity

        })

from collections import defaultdict

table = defaultdict(list)

for r in results:

    table[r["layer"]].append(r)

for layer in table:

    print("="*80)

    print("Layer",layer)

    for x in sorted(table[layer],key=lambda y:y["alpha"],reverse=True):

        print(
            x["alpha"],
            round(x["similarity"],3)
        )