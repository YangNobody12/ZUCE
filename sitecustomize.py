"""
Auto-loaded sitecustomize for all Python subprocesses on Windows.
1. Fixes SSL Certificate Store bug (ASN1: NOT_ENOUGH_DATA).
2. Mocks Unix 'resource' module for Windows EvalPlus subprocess evaluation.
"""
import sys
import types
import certifi
import ssl

# 1. Safe SSL
orig_load = ssl.SSLContext.load_default_certs
def safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        orig_load(self, purpose)
    except Exception:
        self.load_verify_locations(cafile=certifi.where())
ssl.SSLContext.load_default_certs = safe_load_default_certs

# 2. Mock 'resource' module for Windows EvalPlus execution
if "resource" not in sys.modules:
    resource = types.ModuleType("resource")
    resource.RLIMIT_AS = 0
    resource.RLIMIT_DATA = 0
    resource.RLIMIT_STACK = 0
    resource.RLIMIT_NOFILE = 0
    resource.RLIMIT_CPU = 0
    resource.setrlimit = lambda *args, **kwargs: None
    resource.getrlimit = lambda *args, **kwargs: (0, 0)
    sys.modules["resource"] = resource
