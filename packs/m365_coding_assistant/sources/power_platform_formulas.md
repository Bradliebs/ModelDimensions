# Power Platform and PowerApps Formula Examples

Source: assistant local field notes
Version: local-notes-2023
Domain: microsoft
Authority: reputable
Staleness: stale (connectors and formula surface change frequently; verify against current Power Platform docs)

## Key facts

- Power Fx is the low-code formula language used in Power Apps canvas apps; it is
  declarative and spreadsheet-like, recalculating values when their inputs
  change.
- Common functions include `Filter`, `Search`, and `LookUp` for retrieving rows
  from a data source, and `Patch` for creating or updating a record.
- A delegation warning appears when a formula cannot be evaluated at the data
  source and Power Apps would otherwise process only the first 500 (default) or
  up to 2000 rows locally.
- Power Platform DLP policies separate connectors into Business, Non-Business,
  and Blocked groups, and a flow or app cannot mix Business and Non-Business
  connectors in the same resource.

## Examples

- Filter a gallery to active items for the current user:
  `Filter(Tasks, Status = "Active" && Owner.Email = User().Email)`.
- Update a record safely with `Patch`:
  `Patch(Tasks, ThisItem, { Status: "Done", ClosedOn: Now() })`.
- Look up a single value:
  `LookUp(Accounts, AccountId = varSelectedId, AccountName)`.

## Known limitations

- These formula examples are from an older Power Platform release and may not
  reflect the current delegation limits or connector classifications; treat the
  delegation row count and connector groupings as illustrative only.
- `Search` is case-insensitive but is often non-delegable on some data sources,
  which silently caps the rows it scans.

## Targeted recall claims (v2.5A)

Power Fx delegation limits to watch for when filtering large data sources: a delegation warning means the filter is evaluated locally over only the first 500 (up to 2000) rows, so filtering a large data source can silently miss rows; watch for non-delegable functions such as Search when filtering large data sources.
