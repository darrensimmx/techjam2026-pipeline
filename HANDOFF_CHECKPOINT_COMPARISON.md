# Cross-Encoder Checkpoint Comparison - Work in Progress

## Current Status: READY TO RUN

### ✅ Completed
1. **Branch created**: `feature/cross-encoder-checkpoint-comparison`
2. **Script created**: `bakeoff/part4_checkpoint_comparison.py`
   - Tests 5 models at depth=50: TinyBERT-L2, MiniLM-L6, MiniLM-L12, mMiniLM-L12, DistilRoBERTa
   - Fixed cache serialization bug (tuple keys → JSON string keys)
   - Per-model caching for efficient re-runs
   - Shipping criteria evaluation
3. **Data downloaded**: `data/catalog.jsonl` (58 MB, 50,000 products)
4. **Trajectories generated**:
   - `bakeoff/trajectories-current.json` (3.3 MB)
   - `bakeoff/trajectories-legacy.json` (3.6 MB)
5. **Virtual environment setup**: `.venv/` with torch (CPU) and sentence-transformers

### 🔄 Next Steps
1. **Run the comparison** (30-60 minutes):
   ```bash
   cd techjam2026-pipeline
   .venv/Scripts/python bakeoff/part4_checkpoint_comparison.py
   ```

2. **Results will be saved to**:
   - `bakeoff/results-checkpoint-comparison.json` - Full JSON output
   - Terminal output with comparative table and recommendations

3. **After completion**:
   - Commit results file: `git add bakeoff/results-checkpoint-comparison.json`
   - Update documentation:
     - `techjam2026-pipeline/docs/todo.md` - Item 4 with checkpoint decision
     - `techjam2026-docs/features/retrieval-rerank/README.md` - Update axis 3
   - Push branch: `git push -u origin feature/cross-encoder-checkpoint-comparison`
   - Create PR (optional)

### 📁 File Locations
- **Script**: `C:\Users\chowr\OneDrive\Desktop\Code\techjam2026-pipeline\bakeoff\part4_checkpoint_comparison.py`
- **Plan**: `C:\Users\chowr\.claude\plans\keen-baking-walrus.md`
- **Virtual env**: `C:\Users\chowr\OneDrive\Desktop\Code\techjam2026-pipeline\.venv\`
- **Catalog**: `C:\Users\chowr\OneDrive\Desktop\Code\techjam2026-pipeline\data\catalog.jsonl`

### ⚠️ Known Issues (Fixed)
- **Cache serialization bug**: Fixed. Tuple keys `(query, asin)` now serialize as `"query|||asin"` strings.
- **First run crashed**: Script has been fixed. Models are already downloaded to HuggingFace cache.

### 🎯 Expected Output
```
Model                           Tech      Delta    CI Excl.  Hit@10   MRR      s/turn   Size(MB)
K0  BM25 baseline              0.692586    --       --       0.800   0.5256    0.00      --
K1a tinybert-l2                0.XXXXX   +0.XXXX   Yes/No   0.XXX   0.XXXX    X.XX      ~17
K1b minilm-l6 (baseline)       0.708912  +0.0173   No       0.820   0.5254    1.25      91.0
K1c minilm-l12                 0.XXXXX   +0.XXXX   Yes/No   0.XXX   0.XXXX    X.XX      ~135
K1d mminilm-l12                0.XXXXX   +0.XXXX   Yes/No   0.XXX   0.XXXX    X.XX      ~110
K1e distilroberta              0.XXXXX   +0.XXXX   Yes/No   0.XXX   0.XXXX    X.XX      ~250

SHIPPING CRITERIA: [models that pass all 4 criteria]
RECOMMENDED: [best model with justification]
```

### 📝 Git Status
```bash
On branch feature/cross-encoder-checkpoint-comparison
Changes to be committed:
  new file:   bakeoff/part4_checkpoint_comparison.py
  new file:   HANDOFF_CHECKPOINT_COMPARISON.md
```

## Timeline
- **Setup**: 1 hour (completed)
- **Test run**: 30-60 minutes (pending)
- **Analysis & documentation**: 30-60 minutes (pending)
- **Total remaining**: ~1-2 hours

## Context
This addresses `techjam2026-docs/project/open-questions.md` Item 4, axis 3: "Which cross-encoder — never measured." The baseline model (MiniLM-L6) was tested but no checkpoint comparison existed.
