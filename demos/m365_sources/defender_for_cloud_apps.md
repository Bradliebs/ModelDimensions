# Microsoft Defender for Cloud Apps

Source: consultant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- Defender for Cloud Apps is a cloud access security broker that gives
  visibility into cloud app usage and control over data in those apps.
- Cloud Discovery analyses traffic logs to surface shadow IT, ranking discovered
  apps by a risk score so unsanctioned apps can be reviewed.
- App connectors use provider APIs to inspect data already in sanctioned apps
  such as SharePoint, Exchange, and third-party SaaS.
- Conditional Access App Control proxies sessions in real time, allowing
  policies that block download or restrict copy and paste during a risky session.
- Policies can detect anomalies such as impossible travel, mass download, and
  activity from a risky IP address, and can alert or take automated action.

## Decisions and assumptions

- Decision: connect Microsoft 365 first through the app connector before adding
  third-party SaaS, so the highest-value data is covered early.
- Assumption: sign-in risk signals come from Microsoft Entra ID Protection and
  feed session policies, so identity protection is already in place.
- Decision: start anomaly policies in monitor mode and review alerts weekly
  before enabling automatic session blocking.

## Known limitations

- Cloud Discovery depends on traffic logs; without a log source or firewall
  integration, shadow IT visibility is incomplete.
- Session control through Conditional Access App Control requires the session to
  be routed through the reverse proxy, which can affect a few apps' behaviour.
- App connector coverage varies by provider, so not every third-party app
  exposes the same depth of API control.

## Examples

- Example: an impossible-travel alert fires when a user signs in from two distant
  countries within an hour, and the session is flagged for review.
- Example: a session policy blocks download of Confidential files when a user
  connects from an unmanaged device.
