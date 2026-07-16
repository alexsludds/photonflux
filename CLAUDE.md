## Linear Integration

- Fetch issues using the Linear MCP tool.
- Always read the parent issue (if one exists) for full context.
- If a description references a spec file, read it before implementing.
- Set issue status to **In Progress** when starting,
  **In Review** after PR creation.

## Branching

Branch format: `<prefix>/<issue-id-lowercase>-<slug>`
- `feature/` for features
- `fix/` for bugs
- `cleanup/` for tech debt

Example: `feature/gra-12-add-supabase-sync`

## Commits

- Format: `<summary> (<ISSUE-ID>)` e.g. `Add Supabase sync (GRA-12)`
- Never commit code that doesn't build. Run `bun run build` first.

## Pull Requests

Create with `gh pr create`. PR body must include:
- Summary of changes
- Verification: `bun run build` result, files changed
- Link to the Linear issue

## Self-Review (required before pushing)

After implementation, launch a sub-agent to review the diff:
- Check for bugs, dead code, security issues, over-engineering