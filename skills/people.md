---
name: people
description: Identity resolution across NinjaOne people resources
agents: people
---
# People and identity resolution

- Search all relevant people categories when the request says user/person and the category is unclear.
- Keep end users, console users, technicians and contacts as distinct entity types.
- Deduplicate the same underlying record only when the API identity and account type establish equivalence; never merge records solely because names or numeric IDs match.
- Show a useful distinguishing attribute such as email and category when multiple candidates match.
- Ask for one unique candidate before preparing an identity-sensitive change.
- Do not infer a creator, manager or role membership when the API response does not expose it.
