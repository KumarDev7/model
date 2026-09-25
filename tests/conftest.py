import os

# Bit-exact GPU runs (resume test): XLA's GPU scatter-adds are otherwise
# non-deterministic. No effect on CPU.
if "--xla_gpu_deterministic_ops" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()
