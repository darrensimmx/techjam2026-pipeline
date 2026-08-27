# Data

## `public_set.jsonl`

200 labeled dev sessions, copied verbatim from the competition repo
(`TechJam2026/techjam-conversational-search`). Never edit — it's the local
dev/eval signal.

## `catalog.jsonl.gz` / `catalog.jsonl`

The real 50,000-row catalog, vendored into this repo as `catalog.jsonl.gz`
(19MB, checksum-verified against the competition repo's published
`SHA256SUMS` at the `participant-kit` release tag) so nobody has to leave
this repo to fetch it. `catalog.jsonl` itself (60MB decompressed) is
`.gitignore`d — generate it locally with:

```bash
gzip -dk data/catalog.jsonl.gz
```

`scripts/benchmark.py` does this automatically if `catalog.jsonl` is missing.

To re-verify the vendored copy hasn't drifted from the organizer's release:

```bash
sha256sum -c data/SHA256SUMS --ignore-missing
```

Never commit the decompressed `catalog.jsonl` itself -- only the `.gz`.
