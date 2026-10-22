# Harmful Factuality Hallucination (HFH)

Code for **Harmful Factuality: LLMs Correcting What They Shouldn't**.

This repository was reconstructed after repository loss using the archived HFH README and the author's `HelpfulHarm` development code. The original six commits, exact sampled dataset, experiment outputs and framework image were not recovered. See [RECOVERY.md](RECOVERY.md) for source mappings and validation limits.

## Project Entry Points

- Main experiment runner: `run_tasks.py`
- Defense experiments: `defense_experiment.py`
- QA generation (Appendix B): `qa_generator.py`

## Folder Structure

- `Models/`: OpenAI/Azure, Gemini, Llama and Qwen clients
- `data/`: `download_wikientities.py`; generates `sampled_wikientities.json`
- `perturbation/`
  - `gep.py`: Gaussian Embedding Perturbation (soft perturbation)
  - `lier.py`: LLM-Instructed Entity Replacement (hard perturbation)
  - `entity_selector.py`: Head/Body/Tail and theme-related selection
- `evaluation/`: task prompts, JSON input and entity replacement
- `metrics/classifier.py`: automated entity-presence evaluation
- `assets/`: status of the unavailable original framework image

## Requirements

Python 3.10 or later:

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` for OpenAI, or `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and `OPENAI_API_VERSION` for Azure. Model names default to deployment names; use variables such as `AZURE_DEPLOYMENT_GPT_4O` or `AZURE_DEPLOYMENT_GPT_4_1` for custom deployments. See [.env.example](.env.example); export the variables in your shell (the file is not loaded automatically).

For Gemini, set `GEMINI_API_KEY` or place the plain key in `Models/.key`. Keep credentials outside version control. Local Llama/Qwen models require their weights, adequate memory, and any applicable Hugging Face access approval.

## Data Preparation

Download, sample and enrich WikiEntities:

```bash
python data/download_wikientities.py --output data/sampled_wikientities.json
```

The script generates `summary` and `in_sum_entities` using Gemini. `--model` defaults to `GEMINI_MODEL` or `gemini-flash-latest`; record an explicit model and seed for your experiments. This creates a **new** sample, rather than reproducing the unavailable original 1,000 records. Summary enrichment invokes the configured provider and incurs its normal API charges.

To prepare an existing JSON sample without model calls:

```bash
python data/download_wikientities.py --input /path/to/sample.json \
  --output data/sampled_wikientities.json --skip-enrichment
```

Input uses a JSON array of records with `text` and `entities`. Entity lists may also be serialized list strings, as in WikiEntities. Positional groups use the first occurrence of each entity: the first 25% of text is Head, the middle 50% Body, and the final 25% Tail. The legacy theme proxy is `in_sum_entities`, the entities present in the clean summary.

If you already have perturbed data, place it at `data/perturbed_data.json`. Each record needs `entity_pairs`, containing `[original, perturbed]` pairs, plus the original `text` or `perturbed_text`.

## Usage

### 1. Soft perturbation (GEP)

```bash
python -m perturbation.gep --input data/sampled_wikientities.json \
  --output data/perturbed_gep.json --strength 0.1 --seed 42
```

The original BERT mean entity embedding, CLS vocabulary embeddings, Gaussian noise and cosine nearest-word selection are retained. The embedding cache is generated locally in `Cache/Soft/`. Python API: `perturb_wiki_entity(dataframe, save_path=None, perturbation_strength=0.1)`.

### 2. Hard perturbation (LIER)

```bash
python -m perturbation.lier --input data/sampled_wikientities.json \
  --output data/perturbed_lier.json --model gpt4o
```

Requires the configured LLM provider. The entity-type perturbation prompt comes from `HelpfulHarm/perturb.py`.

### 3. Summarization or rephrase

```bash
python run_tasks.py \
  --input data/perturbed_lier.json \
  --output Results/summary.json \
  --eval-output Results/summary_report.txt \
  --task summary --model gpt4o --entities-column entities
```

Use `--task rephrase` for rephrasing. `--entities-column` also accepts available columns such as `head_entities`, `body_entities`, `tail_entities`, or `in_sum_entities`.

### 4. Defense prompt experiments

```bash
python defense_experiment.py \
  --input data/perturbed_lier.json \
  --output Results/defense.json \
  --eval-output Results/defense_report.txt \
  --task summary --model gpt4o
```

The original defense instruction is appended to the task prompt: use only the context and knowledge in the given text, without using interior knowledge.

### 5. QA pairs (Appendix B)

```bash
python qa_generator.py --input data/perturbed_lier.json \
  --output data/qa.json --model gpt-4.1

python run_tasks.py --input data/qa.json --output Results/qa.json \
  --eval-output Results/qa_report.txt --task qa --model gpt4o
```

`qa_generator.py` creates open questions and two-option closed questions. Use `--qa-type closed` to evaluate the latter. As in the development code, option A is the original entity and B the perturbed entity; this fixed ordering should be considered when interpreting results. The QA metric evaluates the answer entity, rather than unrelated context entities.

## Evaluation and Offline Checks

The four entity-level categories are:

| Original entity present | Perturbed entity present | Category |
| --- | --- | --- |
| Yes | Yes | Dual Presence |
| Yes | No | Harmful Factuality |
| No | Yes | Faithful Adherence |
| No | No | Entity Omission |

The recovered metric uses case-insensitive **literal substring** matching. It does not resolve aliases or semantic equivalents. Failed or empty generations are excluded from the denominator and reported separately; a run with failed records exits unsuccessfully. Records retain their entity-level classifications. Report paths ending in `.json` receive a machine-readable report; other suffixes receive text.

Evaluate previously generated records without loading a model:

```bash
python run_tasks.py --input Results/summary.json --output Results/rechecked.json \
  --eval-output Results/rechecked_report.txt --task summary --eval-only

python -m unittest discover -s tests -v
```

The unit tests use synthetic records and injected test clients. They make no API requests, download no datasets or weights, and do not reproduce the paper's experiments.

## Paper and Acknowledgements

[Harmful Factuality: LLMs Correcting What They Shouldn't](https://aclanthology.org/2026.findings-eacl.46/), Findings of EACL 2026.

[Responsible AI Lab](https://rlab.unt.edu/), University of North Texas.
