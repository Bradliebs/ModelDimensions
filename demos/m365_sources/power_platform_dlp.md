# Power Platform Data Loss Prevention

Source: consultant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- Power Platform DLP policies govern which connectors can be used together in
  Power Apps, Power Automate, and Copilot Studio.
- A policy sorts connectors into three groups: Business, Non-Business, and
  Blocked; connectors in different groups cannot be combined in one app or flow.
- Policies are scoped to environments, and a tenant-level policy can apply across
  all environments unless an environment is excluded.
- The Blocked group prevents a connector from being used at all, which is how a
  high-risk connector is removed from citizen-developer reach.
- DLP policies are evaluated when an app or flow runs, so a newly blocked
  connector combination stops working on the next execution.

## Decisions and assumptions

- Decision: place the SQL Server and HTTP connectors in Business, and place
  consumer connectors such as Twitter and personal Dropbox in Blocked.
- Assumption: separate environments exist for development, test, and production,
  each with its own DLP policy tightening toward production.
- Decision: custom connectors default to the Non-Business group until reviewed.

## Known limitations

- DLP classifies whole connectors, not individual actions, so a connector cannot
  be allowed for read but blocked for write through DLP alone.
- A user building an app is not warned at design time in every surface; some
  violations only surface when the app is saved or run.
- DLP does not inspect the data payload itself; it controls connector
  combinations, not the content flowing through them.

## Examples

- Example: a flow that reads from SharePoint (Business) and posts to a personal
  Twitter account (Blocked) fails to run under the policy.
- Example: excluding a sandbox environment lets makers prototype with consumer
  connectors that production forbids.
