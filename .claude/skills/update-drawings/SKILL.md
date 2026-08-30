---
name: update-drawings
description: Check, refresh, and republish the Sol Pipeline Drawings — the six-plate architecture diagram published as a Claude Artifact. Use when asked to update the drawings, republish the pipeline diagrams, regenerate the architecture diagram, check whether the drawings are stale, fix a "drawings stale" warning, or install the drift-check hook.
---

# Update the Sol Pipeline Drawings

Six plates explaining what this repo does and where its score comes from. Live at
<https://claude.ai/code/artifact/f90b4cf5-668e-4086-852e-144893ffeba6>.

**`docs/pipeline-drawings.html` is the source; the published page is downstream of
it.** Edit the file, then republish from it — never edit the published page and
hope the file catches up.

Everything mechanical goes through one driver:
`.claude/skills/update-drawings/drawings.py`. All paths below are relative to the
repo root; run from there.

## Prerequisites

Python 3.10+ and git. Nothing else — the driver is standard-library only and
`requirements.txt` stays comments-only. `check --score` additionally needs
`data/catalog.jsonl` and the `run-sol` skill.

## Check for drift — do this first, always

```bash
python3 .claude/skills/update-drawings/drawings.py check
python3 .claude/skills/update-drawings/drawings.py check --score   # adds Plate 5, ~15s
python3 .claude/skills/update-drawings/drawings.py check --strict  # exit 1 on drift
```

~70 ms. Compares the drawings against live source: the bm25 column weights, the
six-attribute schedule, the ≤40 term cap, the `[0,100]` clamp, every symbol
Plate 3 names, and whether anything under `watch` moved since the pinned commit.
Writes nothing. Exits 0 unless `--strict`.

## Install the post-commit warning

```bash
python3 .claude/skills/update-drawings/drawings.py install-hook
python3 .claude/skills/update-drawings/drawings.py install-hook --uninstall
```

Writes `.git/hooks/post-commit`. It prints at most one line and **always exits 0** —
it cannot block or fail a commit. Git hooks are not version-controlled, so every
teammate runs this once on their own clone.

Silent when clean, and silent during a rebase, merge, or cherry-pick — otherwise a
20-commit rebase would print 20 warnings. `SOL_SKIP_DRAWINGS_CHECK=1` disables it
for one invocation.

## Refresh what is mechanical

```bash
python3 .claude/skills/update-drawings/drawings.py refresh
```

Rewrites the weight caption in Plate 3 and the commit stamps in the masthead and
footer, then updates `pins.json` to match. Instant. **Prose, SVG geometry and
Plate 5's rendered numbers are not touched** — those need a human, and `refresh`
tells you which ones it left for you.

## Republish

The driver cannot do this step — only the Artifact tool can, and only Claude can
call it.

1. Run `check --score` and read the output. **Do not carry any number from memory
   or from this file.**
2. `Artifact` with `action: "read"` and the `url` from `pins.json`. A publish to an
   artifact this session has not read is refused, and reading it also surfaces any
   version someone else published.
3. Edit `docs/pipeline-drawings.html` in place — the masthead stamp, Plate 5's stat
   block and headroom bars, the footer line, plus whatever `check` flagged.
4. `Artifact` with `file_path: docs/pipeline-drawings.html` and the **`url` from
   `pins.json`**. Omitting `url` publishes a *second* artifact and orphans the link
   the team already has.
   - Do **not** pass `favicon`. The 📐 is Artifact-tool metadata, not markup —
     there is no `<link rel="icon">` anywhere in the source. Passing it again on a
     redeploy does nothing.
   - Do **not** wrap the file in `<!doctype>`/`<html>`/`<head>`/`<body>`. The tool
     injects that skeleton, which is exactly why the source starts at `<title>`.
5. Close the loop: `drawings.py refresh`, then commit the HTML and `pins.json`
   together. **Skip this and the hook warns forever.**

## Gotchas

- **The saved source is a fragment, not a document.** No `<!doctype>`, `<html>`,
  `<head>` or `<body>` — the Artifact tool injects those at publish time. `<title>`
  sitting at top level is deliberate and is how the page gets its name; do not
  "fix" it.
- **The footer sha is never compared to `HEAD`.** That would fire on every commit
  and be ignored within a day. It is a provenance stamp — "drawn from tree X" — and
  the real signal is derived from it: did anything under `watch` change since?
- **Plate 5 is never checked by the hook.** It needs the 60 MB gitignored catalog,
  and a missing catalog scores 0.0 silently (`starter/agent.py:102-105` swallows the
  load failure), so an automatic check would report a fake regression on every fresh
  clone. `check --score` guards on the catalog's presence and says so.
- **An extractor that matches 0 or 2 times is drift, not a pass.** If the code shape
  moves, `check` reports `extractor matched N times` rather than silently succeeding
  forever. That silent-pass is the failure mode that would kill this whole system.
- **`title 6.0 … description 1.0` in Plate 3 is prose, not a literal.** It
  paraphrases `bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)` at
  `starter/retrieval.py:76`, so a weight sweep breaks the caption, not just a number.
- **Symbols are pinned by existence, never by line number.** A line pin makes every
  insertion above it a false positive. The paths in `pins.json` are human breadcrumbs.
- **Plates 1, 2, 4 and 6 cannot be drift-checked at all.** "Recall saturates; ranking
  does not" is an editorial claim. Read them yourself on every republish.
- **`evaluator/` is deliberately outside `watch`.** It is vendored and never edited,
  and `bench.py test` already diffs it against the organizer's copy. Adding it here
  would be noise, not coverage.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `extractor matched 0 times` | The code shape moved — fix the pattern in `drawings.py`, not the pin |
| Hook prints nothing, ever | `install-hook` was never run, or neither `python3` nor `python` is on PATH |
| Hook still warns after republishing | You skipped `drawings.py refresh` — the pins are still on the old commit |
| `cannot verify Plate 5: data/catalog.jsonl is absent` | `python3 .claude/skills/run-sol/bench.py setup` |
| `drawings no longer contain the pinned string` | Someone hand-edited the HTML; re-add the string or update `doc_literals` |
| Published page duplicated in the gallery | A publish omitted `url`. Delete the new one; republish to the original |
