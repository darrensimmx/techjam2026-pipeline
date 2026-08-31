# ZeroEntropy zerank-1-small Testing Branch

## What's New
Added **ZeroEntropy zerank-1-small** (1.7B parameter reranker) to the checkpoint comparison.

## Changes Made
1. ✅ Added `zerank-1-small` to model list (6 models total)
2. ✅ Enabled 50-session sampling for faster testing (~1 hour vs 4+ hours)
3. ✅ Added `trust_remote_code=True` support for ZeroEntropy models

## Current Status
**Comparison is running now** - testing 6 models on 50-session sample:
1. TinyBERT-L2 (ultra-lightweight)
2. MiniLM-L6 (baseline)
3. MiniLM-L12 (enhanced)
4. mMiniLM-L12 (multilingual)
5. DistilRoBERTa (semantic specialist)
6. **zerank-1-small** (ZeroEntropy 1.7B reranker) ← NEW

## How to Run
```bash
cd techjam2026-pipeline
git checkout feature/add-zeroentropy-zerank-1-small

# Ensure data is ready
ls data/catalog.jsonl  # Should exist (58 MB)
ls bakeoff/trajectories-current.json  # Should exist (3.3 MB)
ls bakeoff/trajectories-legacy.json   # Should exist (3.6 MB)

# Run comparison (uses 50-session sample)
.venv/Scripts/python bakeoff/part4_checkpoint_comparison.py

# Expected runtime: ~1 hour
# Results: bakeoff/results-checkpoint-comparison.json
```

## What to Test
1. **Does zerank-1-small load successfully?**
   - Requires `trust_remote_code=True`
   - Model size: ~1.7B parameters (larger than others)

2. **Performance comparison:**
   - Does it beat MiniLM-L6 baseline?
   - What's the latency cost (s/turn)?
   - Does it pass shipping criteria?

3. **Shipping criteria (per part0-decision-rule.md §3):**
   - (a) CI excludes zero
   - (b) Delta >= +0.020
   - (c) Regress <= 5%
   - (d) Latency <= 1.0s/turn

## Expected Results Format
```
Model                           Tech      Delta    CI Excl.  Hit@10   MRR      s/turn   Size(MB)
K0  BM25 baseline              0.XXXXX     --       --       0.XXX   0.XXXX    0.00      --
K1a tinybert-l2                0.XXXXX   +0.XXXX   Yes/No   0.XXX   0.XXXX    X.XX      ~17
K1b minilm-l6 (baseline)       0.XXXXX   +0.XXXX   Yes/No   0.XXX   0.XXXX    X.XX      91.0
K1c minilm-l12                 0.XXXXX   +0.XXXX   Yes/No   0.XXX   0.XXXX    X.XX      ~135
K1d mminilm-l12                0.XXXXX   +0.XXXX   Yes/No   0.XXX   0.XXXX    X.XX      ~110
K1e distilroberta              0.XXXXX   +0.XXXX   Yes/No   0.XXX   0.XXXX    X.XX      ~250
K1f zerank-1-small             0.XXXXX   +0.XXXX   Yes/No   0.XXX   0.XXXX    X.XX      ~????

RECOMMENDED: [model name] - [justification]
```

## Key Questions
1. **Is zerank-1-small the best performer?**
2. **What's the accuracy vs latency trade-off?**
3. **Should we test the full 200-session dataset if promising?**

## ZeroEntropy Model Details
- **Model**: `zeroentropy/zerank-1-small`
- **Parameters**: 1.7B (2x smaller than flagship zerank-1)
- **License**: Apache 2.0 (commercial use OK)
- **Source**: https://huggingface.co/zeroentropy/zerank-1-small
- **Docs**: https://docs.zeroentropy.dev/models

## Notes
- 50-session sample saves ~75% time but reduces statistical power
- Results are directionally correct for model comparison
- Winner should be re-tested on full 200 sessions for final validation
- Current comparison is running in background on this machine

## Contact
Questions? Check `HANDOFF_CHECKPOINT_COMPARISON.md` for full context.

Last updated: 2026-08-31
