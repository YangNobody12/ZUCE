import time
import torch
import sitecustomize
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from evalplus.data import get_human_eval_plus
from evalplus.sanitize import sanitize

class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, prompt_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check last generated text
        gen_tokens = input_ids[0][self.prompt_len:]
        if len(gen_tokens) < 2:
            return False
        # Decode only the last 15 tokens for speed
        tail = self.tokenizer.decode(gen_tokens[-15:], skip_special_tokens=True)
        for s in self.stop_strings:
            if s in tail:
                return True
        return False

print("Loading model...")
model_path = "Qwen/Qwen2.5-1.5B"
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

dataset = get_human_eval_plus()
tasks = list(dataset.values())[:10] # test 10 tasks

stop_strings = ["\nif __name__", "\ndef ", "\nclass ", "\nassert ", "\nprint(", "\n```"]

t0 = time.time()
print(f"Testing 10 HumanEval problems with StopOnStrings...")

for i, task in enumerate(tasks):
    t_start = time.time()
    prompt = task["prompt"]
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]
    
    stop_crit = StoppingCriteriaList([StopOnStrings(tok, stop_strings, prompt_len)])
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            stopping_criteria=stop_crit,
            pad_token_id=tok.eos_token_id
        )
    
    gen_tokens = out[0][prompt_len:]
    gen_text = tok.decode(gen_tokens, skip_special_tokens=True)
    
    # Post-process
    for s in stop_strings:
        if s in gen_text:
            gen_text = gen_text.split(s)[0]
    
    full_code = prompt + gen_text
    clean_code = sanitize(full_code, entrypoint=task["entry_point"])
    elapsed = time.time() - t_start
    print(f"Task {task['task_id']}: generated {len(gen_tokens)} tokens in {elapsed:.2f}s ({len(gen_tokens)/max(elapsed, 1e-4):.1f} tok/s)")

total_elapsed = time.time() - t0
print(f"\n10 problems finished in {total_elapsed:.2f}s! (Avg: {total_elapsed/10:.2f}s per problem)")
print(f"Projected time for 164 problems: {total_elapsed/10 * 164 / 60:.1f} minutes!")
