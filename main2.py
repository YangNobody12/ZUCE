import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

model.eval()

########################################################################

PROMPT = """
Write a Python function that returns Fibonacci numbers.
"""

########################################################################

def get_logits():

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model(**inputs)

    return outputs.logits[:, -1, :].float()

########################################################################

baseline_logits = get_logits()

########################################################################

def scale_hook(alpha):

    def hook(module, inp, out):

        if isinstance(out, tuple):
            out = out[0]

        return out * alpha

    return hook

########################################################################

alphas = [1.0,0.8,0.6,0.4,0.2,0.0]

results=[]

########################################################################

for layer in range(len(model.model.layers)):

    print("Layer",layer)

    module = model.model.layers[layer].mlp.down_proj

    for alpha in alphas:

        handle = module.register_forward_hook(
            scale_hook(alpha)
        )

        logits = get_logits()

        handle.remove()

        p = F.log_softmax(
            baseline_logits,
            dim=-1
        )

        q = F.softmax(
            logits,
            dim=-1
        )

        kl = F.kl_div(
            p,
            q,
            reduction="batchmean"
        ).item()

        results.append({

            "layer":layer,

            "alpha":alpha,

            "kl":kl

        })

########################################################################

print()

for layer in range(len(model.model.layers)):

    print("="*80)

    print("Layer",layer)

    for alpha in alphas:

        value = [

            x["kl"]

            for x in results

            if x["layer"]==layer

            and x["alpha"]==alpha

        ][0]

        print(

            f"alpha={alpha:.1f}  KL={value:.6f}"

        )