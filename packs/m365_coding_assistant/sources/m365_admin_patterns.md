# Microsoft 365 Admin Patterns

Source: assistant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- Microsoft 365 administration is split across role-scoped admin centres: the
  Microsoft 365 admin center for tenants and users, Exchange admin center for
  mail flow, and the Teams admin center for meetings and policies.
- Least-privilege administration uses scoped roles such as User Administrator,
  Exchange Administrator, and Helpdesk Administrator instead of granting Global
  Administrator, which should be limited to a small break-glass set.
- Conditional Access policies enforce sign-in requirements such as multifactor
  authentication, compliant devices, and trusted locations before granting
  access to Microsoft 365 services.
- Privileged Identity Management (PIM) makes high-privilege roles eligible
  rather than permanently active, requiring just-in-time activation with
  approval and an audit trail.
- Tenant-wide settings such as external sharing, guest access, and self-service
  group creation are governed centrally and cascade to SharePoint and Teams.

## Decisions and assumptions

- Assume a single production tenant with separate break-glass Global Admin
  accounts excluded from Conditional Access lockout.
- Decision: all admin roles above Helpdesk Administrator are made PIM-eligible,
  not permanently assigned.
- Assumption: multifactor authentication is required for every administrator.

## Known limitations

- These are curated local notes, not a substitute for the current Microsoft
  Learn documentation; role names and admin center layouts change over time.
- Conditional Access misconfiguration can lock out administrators, which is why
  break-glass accounts are excluded from the policies.

## Targeted recall claims (v2.5A)

Explaining role-scoped tenant administration and least-privilege admin roles to a new admin: a Microsoft 365 consultant should explain that tenant administration is split into role-scoped admin centres, and that least-privilege means assigning scoped admin roles such as User Administrator or Helpdesk Administrator instead of Global Administrator, which stays a small break-glass set.
