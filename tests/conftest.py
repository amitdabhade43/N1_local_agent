import os

# Tests must never write audit files into the working tree. test_state_and_audit.py
# reconfigures the sink explicitly (and switches it off again afterwards).
os.environ.setdefault("N1_AUDIT_LOG", "off")
