# Microsoft Purview Endpoint DLP

Source: consultant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- Endpoint data loss prevention extends Purview DLP policies to Windows and
  macOS devices that are onboarded to Microsoft Purview.
- Endpoint DLP monitors and controls activities on the device, including copy to
  USB, copy to network share, print, upload to cloud, and paste to a browser.
- A device must be onboarded before endpoint DLP can see its activity; onboarding
  uses the same pipeline as Microsoft Defender for Endpoint.
- Policy actions range from audit only, to warn the user with an override, to
  block the activity outright.
- Endpoint DLP relies on sensitive information types and can also act on items
  that already carry a Purview sensitivity label.

## Decisions and assumptions

- Decision: start every endpoint DLP policy in audit-only mode for two weeks to
  measure false positives before switching to block.
- Assumption: devices are Microsoft Entra joined and managed by Intune, so
  onboarding is pushed automatically rather than configured by hand.
- Decision: USB copy of Confidential-labelled files is blocked, while print is
  only warned, to balance security against day-to-day work.

## Known limitations

- Endpoint DLP only covers onboarded Windows and macOS devices; Linux and mobile
  are out of scope for endpoint DLP.
- The device must be able to evaluate the policy locally, so a brand-new policy
  can take time to sync to the endpoint before it is enforced.
- Some egress paths, such as a virtual machine clipboard, are not fully covered
  and should be treated as a residual risk.

## Examples

- Example: a user copying a file labelled Highly Confidential to a USB drive is
  blocked, and the action is recorded in activity explorer.
- Example: a user uploading an Internal file to a personal cloud account is
  warned and may continue with a business justification.
