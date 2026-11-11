# Git skills to practice with this suite

This folder is ready for a new repository. Nothing is pushed to GitHub automatically.

```powershell
git init -b main
git add README.md pyproject.toml uv.lock .gitignore .env.example src tests projects docs assets .github Dockerfile compose.yaml .dockerignore
git status
git diff --cached --stat
git commit -m "Build agentic AI portfolio with verified local workflows"
git switch -c feature/retrieval-evaluation
```

Before staging, verify `.env`, API tokens, runtime uploads/DBs, caches and virtual environments are ignored. Inspect actual staged content with `git diff --cached`. The public assets are intentional attributed fixtures, not customer data.

## One reviewable change

1. Add a retrieval experiment or fix a real failure.
2. Run `uv run pytest -q` and `uv run ruff check src tests`.
3. Update the relevant README and evaluation evidence.
4. Stage specific files, inspect the diff, commit with a concrete message.
5. Add your chosen remote, push your branch, and create a pull request containing the problem, change, verification and limits.

```powershell
git remote add origin https://github.com/YOUR-ACCOUNT/agentic-ai-portfolio.git
git push -u origin feature/retrieval-evaluation
```

Git is local version control; GitHub is a hosted collaboration service. A local commit does not publish a project. Use pull requests, protected branches and passing CI before merging team changes. Avoid `reset --hard`, force pushes and copied secrets during routine practice. For an already shared bad commit, prefer a reviewable `git revert`.

## Exercise

Create two branches that edit the same README sentence, merge them locally, resolve the conflict while preserving the intended meaning, rerun validation, and inspect `git log --graph --oneline --all`. Tag a release only after verifying the exact commit and built artifact. Explain how you would roll back a container release and its data separately from reverting source code.
