# Shared project memory protocol

These instructions apply to every Codex task working anywhere under this project.

## At the start of every task

Before planning or changing files, read:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/DECISIONS.md`
3. The latest entries in `docs/WORKLOG.md`
4. Any task-specific document linked from `docs/PROJECT_CONTEXT.md`

Treat these files as the shared memory of all Codex tasks in this project. Do not assume
that the current chat contains the latest project state.

## While working

- Inspect existing files before creating parallel implementations.
- Preserve work made by other tasks and users.
- Record a durable decision in `docs/DECISIONS.md` when it changes data semantics,
  interfaces, experiment design, validation, dependencies, or project structure.
- Keep `docs/PROJECT_CONTEXT.md` concise and current. Replace stale status instead of
  accumulating a transcript there.
- Do not copy secrets, credentials, personal data, or large raw outputs into shared memory.

## Before finishing a task

For every task that produces a material result, append a short entry to
`docs/WORKLOG.md` containing:

- date and task/chat title when known;
- what was investigated or changed;
- important findings and assumptions;
- files created or modified;
- verification performed;
- unresolved questions and the recommended next step.

Then update `docs/PROJECT_CONTEXT.md` if the current state, active work, important paths,
or next steps changed. A task is not complete until its reusable results are visible in
these shared files.

## Cross-task communication limits

Codex chats do not automatically receive full transcripts from sibling chats. The shared
files above are the canonical bridge between them. When exact chat history matters, add a
concise handoff to the worklog or save the relevant discussion under `docs/` and link it
from `docs/PROJECT_CONTEXT.md`.

