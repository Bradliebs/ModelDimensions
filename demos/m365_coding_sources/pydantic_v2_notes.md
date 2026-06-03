# Pydantic v2 Notes

Source: assistant local field notes
Version: local-notes-v1
Domain: coding
Authority: reputable
Staleness: review_required (verify against the Pydantic docs before relying on it)

## Key facts

- Pydantic v2 validates data against a typed model and raises a ValidationError
  when the input does not satisfy the declared types.
- The core was rewritten in Rust (pydantic-core), making v2 substantially faster
  than v1 for both validation and serialisation.
- `BaseModel.model_validate(data)` replaces v1's `parse_obj`, and
  `model_dump()` / `model_dump_json()` replace `dict()` / `json()`.
- Validators changed: `@field_validator` replaces `@validator`, and
  `@model_validator` replaces `@root_validator`.
- Settings management moved out to the separate `pydantic-settings` package;
  `BaseSettings` is no longer in the core `pydantic` package.

## Decisions and assumptions

- Decision: use `model_config = ConfigDict(extra="forbid")` so unexpected fields
  are rejected rather than silently ignored.
- Assumption: code targets Pydantic v2.x; v1 method names are treated as legacy
  and flagged on sight.
- Decision: prefer `model_validate` at system boundaries (request bodies, file
  loads) and trust typed objects internally.

## Known limitations

- v1 and v2 APIs are not drop-in compatible; mixing `dict()` and `model_dump()`
  in one codebase is a common migration bug.
- A `ValidationError` reports all failing fields at once, which is helpful but
  can be verbose in logs if not summarised.
- Custom types may need `__get_pydantic_core_schema__`, which is more involved
  than v1's `__get_validators__`.

## Examples

- Example: `User.model_validate({"name": "A", "age": 30})` returns a typed
  `User`, while a missing `age` raises a `ValidationError`.
- Example: `user.model_dump(exclude={"password"})` serialises a model while
  dropping a sensitive field.

## Caution / near-miss

- Near-miss: a service was upgraded to Pydantic v2 but a handler still called the
  v1 `.dict()` method, which silently still existed via a deprecation shim in an
  early release and returned a subtly different shape downstream. Pinning the
  version and replacing `.dict()` with `model_dump()` resolved it.
