# Microsoft Purview Endpoint DLP

Source: assistant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- Endpoint DLP extends Microsoft Purview data loss prevention to devices,
  monitoring and controlling sensitive items on Windows and macOS endpoints.
- It can audit or block activities such as copying a sensitive file to USB,
  uploading to an unsanctioned cloud service, printing, or pasting to a browser.
- Policies match content using sensitive information types, trainable
  classifiers, or sensitivity labels, then apply an action per activity.
- Devices must be onboarded into the Microsoft Purview compliance portal before
  endpoint policies take effect.
- Policy tips can warn a user at the moment of an action, turning a hard block
  into a teachable moment when configured to allow an override with a reason.

## Decisions and assumptions

- Decision: start endpoint policies in audit mode to size the impact before
  switching any activity to block.
- Assumption: endpoints run a supported, onboarded build; unmanaged personal
  devices are handled by conditional access, not endpoint DLP.
- Decision: USB copy of Confidential-labelled files is blocked, while print is
  audited only.

## Known limitations

- Endpoint DLP only governs onboarded devices; an un-onboarded device is
  invisible to the policy.
- Coverage of third-party browsers depends on the Purview extension being
  installed; an unsupported browser is a gap.
- A block decision is evaluated locally, so a device that is offline for a long
  time may act on a stale policy until it syncs.

## Examples

- Example: a policy blocks copying any Highly Confidential file to a USB drive
  and shows the user a policy tip explaining why.
- Example: an audit-mode policy records every upload of credit-card data to a
  personal cloud account without blocking it, to baseline the risk.

## Caution / near-miss

- Near-miss: a block policy was rolled out tenant-wide on day one and stopped a
  finance team from saving to an approved encrypted USB. Audit-first would have
  caught the false positive before it disrupted work.
