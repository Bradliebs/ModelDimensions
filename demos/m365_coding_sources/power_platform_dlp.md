# Power Platform DLP

Source: assistant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- Power Platform data loss prevention policies govern which connectors a flow or
  app may combine, sorting connectors into Business, Non-Business, and Blocked
  groups.
- A connector in the Business group cannot share data with a connector in the
  Non-Business group within the same app or flow; this separation is the core
  control.
- Policies are scoped to environments, so a production environment can carry a
  stricter policy than a personal developer environment.
- The Blocked group prevents a connector from being used at all in scope.
- DLP policies are evaluated at design time and at run time, so an existing flow
  that violates a new policy is suspended.

## Decisions and assumptions

- Decision: SQL Server, Dataverse, and SharePoint sit in the Business group;
  Twitter and personal cloud storage sit in Non-Business.
- Assumption: a default tenant-wide policy blocks unknown custom connectors
  until they are reviewed.
- Decision: the developer environment is exempt from the production policy to let
  experiments run, but cannot publish to production.

## Known limitations

- Connector groups separate data flow, but they do not inspect the actual data;
  a misclassified connector is the most common gap.
- A custom connector must be explicitly classified, otherwise it inherits the
  default group, which may be more permissive than intended.
- DLP applies to connectors, not to arbitrary HTTP calls made inside a connector
  that already passed the group check.

## Examples

- Example: a policy stops a maker from building an app that reads SharePoint
  (Business) and posts to a personal Twitter account (Non-Business).
- Example: moving the custom payroll connector into the Blocked group instantly
  suspends any flow that depended on it.

## Caution / near-miss

- Near-miss: a new tenant-wide policy was published without checking existing
  flows. A finance reconciliation flow combined two now-separated connectors and
  was suspended overnight. Running an impact review against existing flows first
  would have flagged it.
