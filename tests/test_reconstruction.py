import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from evaluation.common import as_list, entity_pairs, prepare_record, read_records, replace_entities, write_json
from evaluation.prompts import DEFENSE, task_prompt
from metrics.classifier import classify_pair, evaluate_records
from perturbation.entity_selector import classify_entities_by_position, entities_in_summary
from perturbation.lier import perturb_records
from qa_generator import generate_questions
from run_tasks import build_parser, run_pipeline
from data.download_wikientities import prepare_records
from Models.my_openai import OpenAIClient

ROOT = Path(__file__).resolve().parents[1]


def sample():
    return {"text": "Ada wrote a note in London.", "entities": ["Ada", "London"],
            "in_sum_entities": ["Ada"], "head_entities": ["Ada"],
            "entity_pairs": [["Ada", "Grace"], ["London", "Paris"]]}


class ReconstructionTests(unittest.TestCase):
    def test_four_categories(self):
        expected = {"ADA and Grace": "both_present", "Ada": "only_original",
                    "Grace": "only_perturbed", "No names": "neither_present"}
        for text, category in expected.items():
            self.assertEqual(classify_pair(text, "Ada", "Grace"), category)

    def test_no_unchanged_metric_pair(self):
        with self.assertRaises(ValueError):
            classify_pair("Ada", "Ada", "ada")

    def test_replacement_is_simultaneous_and_longest(self):
        text, pairs = replace_entities("New York and York; Ada", [
            ["York", "Ada"], ["New York", "Rome"], ["Ada", "Grace"]])
        self.assertEqual(text, "Rome and Ada; Grace")
        self.assertEqual(len(pairs), 3)

    def test_selected_subset(self):
        row = prepare_record(sample(), "head_entities")
        self.assertEqual(row["perturbed_text"], "Grace wrote a note in London.")
        self.assertEqual(row["active_entity_pairs"], [["Ada", "Grace"]])

    def test_missing_subset_fails(self):
        with self.assertRaises(ValueError):
            prepare_record(sample(), "missing")

    def test_absent_entities_excluded(self):
        row = sample()
        row["entity_pairs"].append(["Berlin", "Rome"])
        self.assertEqual(len(prepare_record(row)["active_entity_pairs"]), 2)

    def test_serialized_fields_are_safe(self):
        self.assertEqual(as_list("['Ada']"), ["Ada"])
        with self.assertRaises((ValueError, SyntaxError)):
            as_list("__import__('os').system('false')")

    def test_conflicting_pairs_fail(self):
        row = sample()
        row["entity_pairs"].append(["Ada", "Marie"])
        with self.assertRaises(ValueError):
            entity_pairs(row)

    def test_failure_is_not_entity_omission(self):
        good = {**prepare_record(sample(), "head_entities"), "summary": "Ada"}
        bad = {**good, "summary": None, "generation_error": "Request failed"}
        _, report = evaluate_records([good, bad])
        self.assertEqual(report["failed_records"], 1)
        self.assertEqual(report["total_entity_pairs"], 1)
        self.assertEqual(report["categories"]["only_original"]["percentage"], 100)

    def test_position_groups(self):
        groups = classify_entities_by_position("A" + "-" * 49 + "B" + "-" * 48 + "C", ["A", "B", "C"])
        self.assertEqual(groups, {"head_entities": ["A"], "body_entities": ["B"], "tail_entities": ["C"]})

    def test_summary_theme_proxy(self):
        self.assertEqual(entities_in_summary(["Ada", "London"], "ADA wrote."), ["Ada"])

    def test_data_preparation_with_injected_summary(self):
        result = prepare_records([sample()], lambda *a, **k: "Ada wrote.")
        self.assertEqual(result[0]["in_sum_entities"], ["Ada"])

    def test_lier_prompt_and_output(self):
        row = {"text": "Ada", "entities": ["Ada"]}
        result = perturb_records([row], lambda *a, **k: '{"entity":"Ada","perturbed":"Grace"}')
        self.assertEqual(result[0]["entity_pairs"], [["Ada", "Grace"]])

    def test_lier_wrong_entity_rejected(self):
        with self.assertRaises(ValueError):
            perturb_records([{"entities": ["Ada"]}], lambda *a, **k: '{"entity":"X","perturbed":"Y"}')

    def test_qa_generation_keeps_entity_pair(self):
        records = generate_questions([sample()], lambda *a, **k: "Who wrote a note?")
        self.assertEqual(records[0]["ori_answer"], "Ada")
        self.assertEqual(records[0]["pert_answer"], "Grace")
        self.assertEqual(records[0]["choice_B"], "Grace")

    def test_defense_prompt(self):
        _, system = task_prompt(prepare_record(sample()), "summary", defense=True)
        self.assertIn(DEFENSE, system)

    def test_api_request_without_credentials_or_network(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="text"))])
        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client = OpenAIClient(client=fake)
        self.assertEqual(client.gpt4o("hello"), "text")
        self.assertEqual(calls[-1]["model"], "gpt-4o")
        client.o4_mini("hello")
        self.assertIn("max_completion_tokens", calls[-1])
        self.assertNotIn("temperature", calls[-1])
        self.assertNotIn("top_p", calls[-1])

    def test_local_adapters_greedy_decoding_and_continuation(self):
        import contextlib
        import importlib
        import numpy as np
        torch_stub = SimpleNamespace(no_grad=contextlib.nullcontext)
        transformers_stub = SimpleNamespace(AutoTokenizer=object, AutoModelForCausalLM=object)
        class Tensor:
            shape = (1, 2)
            def to(self, device):
                return self
            def __getitem__(self, item):
                return np.array([[1, 2]])[item]
        class Tokenizer:
            eos_token_id = 0
            def apply_chat_template(self, *a, **kw):
                return Tensor()
            def decode(self, tokens, **kw):
                self.decoded_tokens = list(tokens)
                return "Ada"
        for module_name, class_name in (("Models.my_llama", "LlamaClient"),
                                        ("Models.my_qwen", "QwenClient")):
            with mock.patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}):
                module = importlib.import_module(module_name)
                client = getattr(module, class_name).__new__(getattr(module, class_name))
                client.device = "cpu"
                client.tokenizer = Tokenizer()
                calls = []
                def generate(*a, **kw):
                    calls.append(kw)
                    return np.array([[1, 2, 9]])
                client.model = SimpleNamespace(generate=generate)
                self.assertEqual(client.generate("question", temperature=0), "Ada")
                self.assertEqual(client.tokenizer.decoded_tokens, [9])
                self.assertFalse(calls[-1]["do_sample"])
                self.assertNotIn("temperature", calls[-1])

    def test_empty_deployment_override_uses_model_name(self):
        calls = []
        def create(**kw):
            calls.append(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Ada"))])
        client = OpenAIClient(client=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create))))
        with mock.patch.dict("os.environ", {"AZURE_DEPLOYMENT_GPT_4O": ""}):
            client.gpt4o("hello")
        self.assertEqual(calls[-1]["model"], "gpt-4o")

    def test_summary_pipeline_and_json_report(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            write_json(base / "in.json", [sample()])
            args = build_parser().parse_args(["--input", str(base / "in.json"),
                "--output", str(base / "out.json"), "--eval-output", str(base / "report.json"),
                "--entities-column", "head_entities"])
            rows, report = run_pipeline(args, lambda *a, **k: "Grace wrote a note.")
            self.assertEqual(report["total_entity_pairs"], 1)
            self.assertEqual(report["categories"]["only_perturbed"]["count"], 1)
            self.assertEqual(json.loads((base / "report.json").read_text())["failed_records"], 0)
            self.assertEqual(read_records(base / "out.json")[0]["summary"], rows[0]["summary"])

    def test_closed_qa_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            write_json(base / "in.json", generate_questions([sample()], lambda *a, **k: "Who wrote?"))
            args = build_parser().parse_args(["--input", str(base / "in.json"),
                "--output", str(base / "out.json"), "--eval-output", str(base / "report.txt"),
                "--task", "qa", "--qa-type", "closed"])
            rows, report = run_pipeline(args, lambda *a, **k: "B")
            self.assertEqual(rows[0]["answer"], "Grace")
            self.assertEqual(report["total_entity_pairs"], 1)
            self.assertEqual(report["categories"]["only_perturbed"]["count"], 1)

    def test_failed_generation_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            write_json(base / "in.json", [sample()])
            args = build_parser().parse_args(["--input", str(base / "in.json"),
                "--output", str(base / "out.json"), "--eval-output", str(base / "report.txt")])
            def fail(*a, **k):
                raise RuntimeError("Synthetic provider failure")
            _, report = run_pipeline(args, fail)
            self.assertEqual(report["failed_records"], 1)
            self.assertEqual(report["total_entity_pairs"], 0)

    def test_cli_help_without_models(self):
        commands = [["run_tasks.py"], ["defense_experiment.py"], ["qa_generator.py"],
                    ["data/download_wikientities.py"], ["-m", "perturbation.gep"],
                    ["-m", "perturbation.lier"]]
        for command in commands:
            result = subprocess.run([sys.executable, *command, "--help"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
