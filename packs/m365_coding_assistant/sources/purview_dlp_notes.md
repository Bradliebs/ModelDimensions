# Purview Sensitivity Labels and DLP Notes

Source: assistant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- Microsoft Purview sensitivity labels classify and protect content by applying
  a persistent label that travels with the file or email and can enforce
  encryption, content marking, and access restrictions.
- A sensitivity label taxonomy commonly uses Public, Internal, Confidential, and
  Highly Confidential, where encryption is enabled only at Confidential and
  above so encrypted files do not break third-party scanners.
- Data Loss Prevention (DLP) policies detect sensitive information types such as
  credit card numbers or national identifiers and can block, warn, or audit when
  content is shared outside permitted boundaries.
- Endpoint DLP extends these policies to devices, controlling actions such as
  copying a labelled file to a USB drive, uploading to an unmanaged cloud, or
  printing.
- A sensitivity label is not a retention label; a sensitivity label classifies
  and protects, while a retention label sets how long content is kept.

## Decisions and assumptions

- Decision: DLP policies start in audit mode, then move to block mode after a
  baseline of false positives has been reviewed.
- Assumption: only one sensitivity label applies to an item at a time, and the
  higher-priority label is treated as more restrictive.

## Known limitations

- Auto-labelling for files at rest runs on a schedule and is not instantaneous,
  so a new file may stay unlabelled for some time.
- Encryption applied by a label can block server-side services such as
  eDiscovery search unless the service is explicitly trusted.

## Targeted recall claims (v2.5A)

Purview sensitivity label encryption thresholds: in this pack the encryption threshold is the label tier at which encryption turns on. Encryption is enabled at Confidential and Highly Confidential, while Public and Internal labels stay unencrypted so files do not break third-party scanners.

Designing a Microsoft Purview sensitivity label taxonomy: choose four tiers — Public, Internal, Confidential, and Highly Confidential — and choose the encryption threshold by deciding the lowest tier that enforces encryption, which this taxonomy sets at Confidential and above.

Protecting Highly Confidential files with sensitivity labels while keeping third-party tools working: apply encryption only to the Highly Confidential and Confidential labels, and explicitly trust the third-party tools that must read the files, so protection holds without breaking those tools.
