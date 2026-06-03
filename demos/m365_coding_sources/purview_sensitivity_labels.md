# Microsoft Purview Sensitivity Labels

Source: assistant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- Sensitivity labels classify and protect content in Microsoft 365 by applying a
  persistent label that travels with the file or email.
- A label can enforce encryption, content marking (header, footer, watermark),
  and access restrictions that follow the document outside the tenant.
- Labels are published to users through label policies, which scope which labels
  which users and groups can see.
- Container labels apply to SharePoint sites, Microsoft 365 Groups, and Teams,
  controlling privacy, external sharing, and unmanaged-device access.
- Auto-labelling can apply a label based on sensitive information types, for
  example detecting credit card or national ID patterns at rest or in transit.

## Decisions and assumptions

- Assume a single tenant with a published taxonomy of Public, Internal,
  Confidential, and Highly Confidential.
- Decision: encryption is enabled only on Confidential and above, because
  encrypted files break some third-party data-loss-prevention scanners.
- Assumption: only one sensitivity label applies to an item at a time, and a
  higher-priority label is treated as more restrictive.

## Known limitations

- A sensitivity label and a retention label are different things; a sensitivity
  label does not set a retention period.
- Encryption applied by a label can prevent server-side services such as
  eDiscovery search or co-authoring unless the service is explicitly trusted.
- Auto-labelling for files at rest runs on a schedule and is not instantaneous,
  so a newly created file may stay unlabelled for some time.

## Examples

- Example: a Confidential label adds a "Confidential" footer and restricts access
  to internal users only.
- Example: a container label on a Team sets the group to private and blocks
  sharing with guests.

## Caution / near-miss

- Near-miss: a team assumed a sensitivity label would auto-delete old documents.
  It does not — that is a retention label. Pairing the two without checking the
  difference nearly left regulated data past its disposal date.
