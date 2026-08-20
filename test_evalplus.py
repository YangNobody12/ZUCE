import certifi
import ssl

orig_load = ssl.SSLContext.load_default_certs
def safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        orig_load(self, purpose)
    except Exception:
        self.load_verify_locations(cafile=certifi.where())
ssl.SSLContext.load_default_certs = safe_load_default_certs

from evalplus.data import get_human_eval_plus
data = get_human_eval_plus()
print(f"Total tasks: {len(data)}")
first_key = list(data.keys())[0]
print(f"Task {first_key} keys: {list(data[first_key].keys())}")
print(f"Task {first_key} prompt:\n{data[first_key]['prompt']}")
print(f"Task {first_key} entry_point: {data[first_key]['entry_point']}")
