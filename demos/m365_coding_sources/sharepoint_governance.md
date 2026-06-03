# SharePoint Governance

Source: assistant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- External sharing is controlled at the organisation level and can be tightened
  per site, ranging from "anyone" links down to "only people in your org".
- SharePoint governance covers how sites are created, shared, retained, and
  retired across a tenant, balancing self-service against sprawl.
- Site creation can be restricted so only approved users or a request process can
  provision new sites and the Microsoft 365 Groups behind them.
- Sensitivity labels applied as container labels set a site's privacy, external
  sharing, and unmanaged-device access in one place.
- Site lifecycle policies can detect inactive sites and prompt owners to attest,
  archive, or delete them.

## Decisions and assumptions

- Decision: external sharing defaults to "new and existing guests" tenant-wide,
  and is narrowed to "only people in your org" on sites holding Confidential
  content.
- Assumption: every site has at least two owners so governance attestations are
  never blocked by a single absent owner.
- Decision: self-service site creation is disabled; sites are provisioned through
  a request flow that applies the right container label.

## Known limitations

- A container label sets a site's posture at creation, but changing the label
  later does not retroactively re-permission existing content.
- Inactive-site detection is based on activity signals and can misjudge a site
  that is read often but rarely edited.
- Organisation-level external sharing is a ceiling; a per-site setting can be
  stricter but never more permissive than the tenant ceiling.

## Examples

- Example: a project site is provisioned private with a Confidential container
  label, blocking guest access from the start.
- Example: a lifecycle policy emails owners of a site with no activity for 180
  days and archives it if no one attests.

## Caution / near-miss

- Near-miss: a tenant loosened the organisation-level sharing ceiling to enable
  one partner project, unintentionally allowing "anyone" links on hundreds of
  sites whose per-site setting inherited the default. Scoping the change to a
  single site collection would have contained it.
