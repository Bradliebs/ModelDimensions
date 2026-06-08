"""Reading-engine diagnostics.

Each module here turns one claim from the "Knowledge-Free Reader" roadmap
into a callable function with explicit numeric pass/fail. The wrappers in
``scripts/`` are thin CLIs over these functions; the functions themselves
are pure and unit-testable without a GPU, a model, or the live bank.
"""

from . import acceptance, compute_plan, disjointness, layer_attribution  # noqa: F401
