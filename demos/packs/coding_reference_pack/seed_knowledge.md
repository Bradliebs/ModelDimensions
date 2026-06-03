# Python Coding Reference

A list comprehension builds a new list from an iterable in a single expression.
It reads left to right: the output expression, then a for-clause, then an
optional if-clause that filters items.

A dictionary comprehension builds a dict with `{key: value for item in iterable}`
and may also take a trailing if-clause to filter which keys are produced.

A generator expression looks like a list comprehension but uses parentheses; it
yields items lazily and does not build the whole sequence in memory.

The `with` statement opens a context manager and guarantees its cleanup runs even
if the block raises, which is the standard way to manage files and locks.
