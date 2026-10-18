# Summary
This document lists the literature reviewed for the Repo-Aware Code Generation project and a short summary of how each paper relates to the project.

- [RepoFuse](https://arxiv.org/abs/2402.14323) (arXiv 2402.14323) is Pass 1. It compares signature context against retrieved-snippet context at matched truncation from 256 to 4096 tokens on CrossCodeEval. Its headline result is that fusing the two beats either one alone.
- [Repoformer](https://arxiv.org/abs/2403.10059) (arXiv 2403.10059) is Pass 2. It learns per example whether to retrieve or skip, which is already a cost-aware selector.
- [ProCC](https://arxiv.org/abs/2405.07530) (arXiv 2405.07530) is also Pass 2. It uses a contextual bandit to pick a retrieval strategy per example.