"""
Unified CLI entry point with safe SSL certificate patch and UTF-8 stdout for Windows.
"""
import sys
import os
import certifi
import ssl

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

orig_load = ssl.SSLContext.load_default_certs
def safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        orig_load(self, purpose)
    except Exception:
        self.load_verify_locations(cafile=certifi.where())

ssl.SSLContext.load_default_certs = safe_load_default_certs

if __name__ == "__main__":
    tool = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if tool == "evalplus_codegen":
        from evalplus.codegen import run_codegen
        from evalplus.evaluate import main
        # Run codegen CLI
        import fire
        from evalplus.codegen import main as codegen_main
        fire.Fire(codegen_main)
    elif tool == "evalplus_evaluate":
        from evalplus.evaluate import main as eval_main
        import fire
        fire.Fire(eval_main)
    elif tool == "lm_eval":
        from lm_eval.__main__ import cli_evaluate
        cli_evaluate()
    else:
        print(f"Unknown tool: {tool}")
        sys.exit(1)
