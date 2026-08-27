# Data

## `public_set.jsonl`

200 labeled dev sessions, copied verbatim from the competition repo
(`TechJam2026/techjam-conversational-search`). Never edit — it's the local
dev/eval signal.

## `catalog.jsonl` — you need to fetch this yourself

Not committed here (50,000 rows, distributed as a GitHub Release asset, not a
repo file). This sandbox's GitHub access is read-only git-clone for that repo,
which doesn't cover release downloads — so this had to be left for a human
with normal GitHub access:

```bash
# from https://github.com/TechJam2026/techjam-conversational-search/releases
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify against the published `SHA256SUMS` file. Nothing in `starter/retrieval.py`
or the two dev CLIs works without this file in place. Tests don't need it —
they run against `tests/fixtures/catalog.jsonl`, a small synthetic catalog.

Never commit the real `catalog.jsonl` — it's in `.gitignore`.
