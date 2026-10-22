# Reconstruction provenance

Reconstruction date: 2026-10-22.

## Sources

- Archived public HFH homepage: https://web.archive.org/web/20260321031501/https://github.com/ResponsibleAILab/Harmful-Factuality-Hallucination
- Development source: `FatShion-FTD/HelpfulHarm`, commit `9c304a07bac3fb70ef71f4a81273863035d41c15`.
- The repository owner identified HelpfulHarm as the source of most of the lost HFH implementation and authorized this reconstruction.

## File and behavior mappings

| Restored component | Development source | Treatment |
| --- | --- | --- |
| `perturbation/gep.py` | `perturb.py`: `perturb_wiki_entity`, `get_vocab_embeddings`, `get_similar_words` | Core function bodies retained; lazy imports, safe cache loading, CLI and seeded invocation added |
| `perturbation/lier.py` | `perturb.py`: `perturb_entity_with_openai` | Original type-preserving JSON prompt retained; public CLI and strict JSON validation rebuilt |
| `perturbation/entity_selector.py` | `wikiEntity_exp.py`: position classification and summary entity selection | Same 25/50/25 split, first occurrence and literal clean-summary presence |
| `run_tasks.py`, `evaluation/prompts.py` | `run_experiments.py`: summary/rephrase generation | Original task prompts and model choices; common orchestration rebuilt |
| `defense_experiment.py` | `run_experiments.py`: `def_only` prompt | Original defense instruction retained |
| `metrics/classifier.py` | `run_experiments.py`: `evaluate_entity_presence` | Original four lexical-presence categories; failure and denominator handling made explicit |
| `qa_generator.py` | `QA_wikiEntity.py`: answer selection and open/closed question prompts | Prompts, first eligible summary entity and A/B entity choices retained; configurable paths rebuilt |
| QA task evaluation | `run_qa_experiment.py` | Context/question generation and A/B response interpretation rebuilt; answer-pair presence metric retained |
| `Models/my_llama.py`, `Models/my_qwen.py` | Files with the same names | Source retained; continuation decoded by token offset to avoid unreliable prompt string slicing |
| `Models/my_openai.py` | File with the same name | Method names retained; embedded Azure endpoint/key removed; environment-based provider configuration rebuilt |
| `data/download_wikientities.py` | `wikiEntity_exp.py`, `complete_wikientity_dataset.py`, archived README | WikiEntities schema and position selection retained; Gemini enrichment added to match archived usage |
| `README.md`, directory layout and CLI flags | Archived HFH homepage/README | Recovered content organized with current configuration and reconstruction notes |
| `LICENSE` | Archived license identification | Standard Apache-2.0 text restored; unavailable original copyright notices were not invented |

## Explicit reconstruction decisions

- The original six HFH commit objects were unavailable. This repository starts a new history; it does not claim to reproduce those commits.
- Private development history, local settings, sync scripts, result notes, API credentials and experiment artifacts are excluded.
- Entity replacement uses simultaneous longest-match substitution to avoid cascading one replacement into another. Missing requested entity columns are errors rather than silent fallbacks to all entities.
- Unchanged pairs, entities not actually replaced in the input, and failed generations are excluded from evaluation. The four category percentages are entity-level.
- LIER invalid responses fail clearly instead of silently substituting the original entity. GEP retains the original per-record failure flag.
- The QA generation scripts do not reconstruct the legacy clean-question ROUGE/correctness prefilter or any lost prefiltered QA dataset. Generated questions should be checked before comparison with published QA numbers.
- The original 1,000-row sample, exact model snapshots, random seeds, generated questions, results and `assets/SCH-category.jpg` were unavailable. No synthetic experiment results are supplied. The downloader generates a fresh sample with an explicit seed.
- Requirements are compatibility ranges for this reconstruction; the original dependency lock file was unavailable.

## Validation scope

Offline tests cover the four categories, subset selection, replacement overlap/cascading, failure exclusion, report totals, data preparation, open/closed QA, LIER parsing, provider request construction and CLI entry points. All Python files are syntax-checked. Test generators and synthetic records are isolated in `tests/`.

No paid model request, real dataset download, BERT weight download or GPU inference is part of these checks. Exact reproduction of the paper's measurements requires the original data and experimental settings.
