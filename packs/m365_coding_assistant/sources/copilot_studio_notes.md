# SharePoint, Teams, and Copilot Studio Notes

Source: assistant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- SharePoint governance covers how sites are created, shared, retained, and
  retired across a tenant, balancing self-service against sprawl; external
  sharing is set at the organisation level and can be tightened per site from
  "anyone" links down to "only people in your org".
- Microsoft Teams is backed by SharePoint and Exchange: each team has a
  SharePoint site for files and a Microsoft 365 Group for membership, so Teams
  governance inherits the underlying site and group controls.
- Copilot Studio builds custom agents grounded in connected knowledge sources;
  the agent does not browse the live web unless a web source is explicitly
  configured, and generative answers are only as current as the connected
  knowledge.
- An agent cannot exceed the permissions of its authentication context, so a
  user without access to a document will not get answers grounded in it.
- Response quality degrades when knowledge sources are large and unstructured,
  so curating the grounding corpus matters more than its size.

## Decisions and assumptions

- Decision: new SharePoint site creation goes through a request-and-approval
  process rather than open self-service.
- Assumption: Copilot Studio agents are scoped to a single curated knowledge
  source per use case to keep grounding quality high.

## Known limitations

- These notes are curated locally; Copilot Studio capabilities and connector
  availability change frequently and should be checked against current docs.
- Teams and SharePoint permission inheritance can surprise users when a file's
  effective access differs from the team's membership.
