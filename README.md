# Harmful Factuality Hallucination (HFH)

This repo contains the code from "Harmful Factuality Hallucination: LLMs Correcting What They Shouldn't"

![framework](https://github.com/user-attachments/assets/bdc2838c-5b5b-40fb-b2b0-ab8b069ed50f)

# Project Entry Points

- Main experiment runner: `run_tasks.py`
- Defense experiments: `defense_experiment.py`
- QA generation (Appendix B): `qa_generator.py`

# Folder Structure

- `data/`
  - `sampled_wikientities.json`
  - `download_wikientities.py`
- `perturbation/`
  - `gep.py`: Gaussian Embedding Perturbation (soft perturbation)
  - `lier.py`: LLM-Instructed Entity Replacement (hard perturbation)
  - `entity_selector.py`: Head/Body/Tail and theme-related selection
- `evaluation/`
  - `prompts.py`: prompt templates (summary, rephrase, QA)
- `metrics/`
  - `classifier.py`: automated entity-presence evaluation

# Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

# Data Preparation

Download and sample the dataset:

```bash
python data/download_wikientities.py --output data/sampled_wikientities.json
```

This script generates `summary` and `in_sum_entities` using Gemini.
Place your Gemini API key at: `Models/.key`

If you already have perturbed data, place it as:

```text
data/perturbed_data.json
```

# Usage

1) Soft perturbation (GEP)

```bash
python -m perturbation.gep
```

(Use the returned dataframe in your pipeline; see code for API.)

2) Hard perturbation (LIER)

```bash
python -m perturbation.lier
```

(Requires OpenAI API access; see code for API.)

3) Run summarization or rephrase task

```bash
python run_tasks.py \
  --input <input_json> \
  --output <output_json> \
  --eval-output <report_txt> \
  --task summary \
  --model gpt4o \
  --entities-column entities
```

4) Run defense prompt experiments

```bash
python defense_experiment.py \
  --input <input_json> \
  --output <output_json> \
  --eval-output <report_txt> \
  --task summary \
  --model gpt4o
```

5) Generate QA pairs (Appendix B)

```bash
python qa_generator.py \
  --input <perturbed_json> \
  --output <qa_json> \
  --model gpt-4.1
```

# Acknowledgements

Responsible AI Lab, University of North Texas

# Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{li-etal-2026-harmful,
    title = "Harmful Factuality: {LLM}s Correcting What They Shouldn{'}t",
    author = "Li, Mingchen  and
      Zhang, Hanzhi  and
      Fan, Heng  and
      Ding, Junhua  and
      Feng, Yunhe",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-eacl.46/",
    doi = "10.18653/v1/2026.findings-eacl.46",
    pages = "896--912",
    ISBN = "979-8-89176-386-9"
}
```
