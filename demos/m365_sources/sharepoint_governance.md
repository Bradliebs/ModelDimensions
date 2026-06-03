# SharePoint Governance

Source: consultant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- SharePoint governance defines how sites are provisioned, secured, shared, and
  retired across the tenant.
- Site provisioning can be controlled so that creating a new site or Microsoft
  365 Group follows an approved template and naming convention.
- External sharing is set at two levels: an organization-wide ceiling and a
  per-site setting that can be equal to or more restrictive than the tenant.
- Access is managed through SharePoint groups and Microsoft 365 Group
  membership; direct per-item permissions are discouraged because they are hard
  to audit.
- Site lifecycle policies can archive or delete inactive sites, reducing sprawl
  and the surface area for oversharing.

## Decisions and assumptions

- Decision: disable unmanaged site creation by end users and route requests
  through a provisioning process with an owner and a purpose.
- Assumption: every site has at least two owners so a site is never orphaned when
  one owner leaves.
- Decision: external sharing defaults to "existing guests only" at the tenant,
  and only specific collaboration sites are raised to "new and existing guests".

## Known limitations

- Tightening tenant-level external sharing cannot be overridden upward by a site;
  a site can only be equal or more restrictive.
- Breaking permission inheritance on a single library or item creates unique
  permissions that are easy to lose track of during an audit.
- Governance policy is only as good as its enforcement; without provisioning
  controls, users can still create groups from Teams or Outlook.

## Examples

- Example: a new project site is created from an approved template with a naming
  prefix and two assigned owners.
- Example: an inactive site with no activity for one year is flagged by a
  lifecycle policy for archive or owner attestation.
