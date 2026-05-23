# Workaround: pyarrow 24.0.0 and torch 2.5.1+cu121 have a DLL conflict on
# Windows when pyarrow's C extension loads AFTER torch/CUDA. Pre-importing
# pyarrow here ensures it loads before torch regardless of import order.
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass
