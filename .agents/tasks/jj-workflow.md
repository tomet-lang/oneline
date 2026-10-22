# jj-workflow

Goal: Standard Jujutsu (`jj`) workflow guide and configuration for the `oneline` repository.

## Context

`oneline` uses Jujutsu (`jj`) co-located with Git for version control.
This ensures reproducible, branchless, conflict-free commit management while maintaining 100% compatibility with the GitHub remote (`git@github-kakusika:tomet-lang/oneline.git`).

## Gotchas & Rules

1. **Always use `--no-pager`**:
   In non-interactive or agentic shells, `jj` commands that launch a pager (e.g. `less`) can cause the shell to wait indefinitely. Always append `--no-pager`:
   ```bash
   jj status --no-pager
   jj log --no-pager -n 5
   jj diff --no-pager
   ```
2. **Author and Privacy Constraints**:
   - Strictly NO personal identifiers or paths (e.g. usernames or home directory paths) in commit messages or author info.
   - Author name: `kakusika`
   - Author email: `329925104+kakusika@users.noreply.github.com`
   - Verify config with: `jj config list` or `git config user.name && git config user.email`.
3. **Commit & Push Workflow**:
   ```bash
   # 1. Describe working copy changes
   jj describe -m "feat: your descriptive commit message" --no-pager

   # 2. Advance the main bookmark to current revision (@)
   jj bookmark set main -r @ --no-pager

   # 3. Push to remote
   jj git push --no-pager

   # 4. Open a clean working copy for the next task
   jj new --no-pager
   ```
4. **Synchronizing with Git**:
   If changes are made via `git` directly or remote updates arrive:
   ```bash
   jj git import --no-pager
   ```

## Checklist

- [x] Initialized co-located `.jj` repository.
- [x] Track remote bookmark `main@origin`.
- [x] Verified commit and push workflow using `jj`.
