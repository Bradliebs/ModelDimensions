# Copilot Studio Agent Patterns

Source: assistant local field notes
Version: local-notes-v1
Domain: microsoft
Authority: reputable
Staleness: review_required (verify against Microsoft Learn before client delivery)

## Key facts

- Copilot Studio builds custom agents that combine topics, generative answers,
  and actions over enterprise data.
- Knowledge sources ground an agent's generative answers; sources include public
  websites, SharePoint, Dataverse, and uploaded documents.
- Topics define authored conversation paths with trigger phrases, while
  generative orchestration lets the agent choose actions and topics dynamically.
- Actions let an agent call connectors, Power Automate flows, or REST APIs to do
  real work beyond answering questions.
- Authentication can be configured so the agent acts on behalf of the signed-in
  user, respecting that user's existing permissions on the grounded data.

## Decisions and assumptions

- Decision: enable generative orchestration only after the authored topics are
  stable, so behaviour stays predictable during early testing.
- Assumption: the agent is published to Microsoft Teams as the primary channel
  rather than a public website, to keep grounding data inside the tenant.
- Decision: every action that writes data requires an explicit confirmation step
  in the conversation before it runs.

## Known limitations

- Generative answers are only as current as the connected knowledge source; the
  agent does not browse the live web unless a web source is configured.
- An agent cannot exceed the permissions of its authentication context, so a
  user without access to a document will not get answers grounded in it.
- Response quality degrades when knowledge sources are large and unstructured, so
  curating the grounding corpus matters.

## Examples

- Example: an HR agent answers leave-policy questions grounded in a SharePoint
  library and hands off to a person for anything it cannot answer.
- Example: an IT agent triggers a Power Automate flow to reset a password after
  the user confirms the request in the chat.

## Caution / near-miss

- Near-miss: an agent was published to a public website with a Dataverse
  knowledge source still attached. Because the public channel had no user
  authentication, every visitor would have shared one service identity's access.
  Catching it before launch avoided exposing internal records.
