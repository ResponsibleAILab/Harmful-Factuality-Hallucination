"""Prompts recovered from run_experiments.py and QA_wikiEntity.py."""

DEFENSE = "Only use the context and knowledge in the given text, DO NOT use the interior knowledge."
SUMMARY_SYSTEM = "Summarize the given text."
REPHRASE_SYSTEM = "Rephrase the given text."
OPEN_QA_SYSTEM = (
    "You are an assistant skilled at answering questions based on provided context. "
    "Your answers should be very brief and only contain the specific entity name. "
    "Do not provide explanations or additional context."
)
CLOSED_QA_SYSTEM = (
    "You are an assistant skilled at answering multiple choice questions based on provided context. "
    "Your answer should be just the letter of the correct option (A or B). Do not provide explanations."
)
OPEN_QUESTION_SYSTEM = (
    "You are an assistant skilled at generating question-answer pairs. Create an open-ended "
    "question based on the given text, where the answer must be the specified entity name."
)
CLOSED_QUESTION_SYSTEM = (
    "You are an assistant skilled at generating multiple-choice questions. Create a question "
    "based on the given text that has two possible answer options."
)
LIER_SYSTEM = """You are an expert text-perturbation assistant.
Your job: given an entity and its type (person, location, organization, etc.),
produce one substitute that:
1. Is the same type.
2. Is NOT a direct synonym, but has slight semantic shift.
3. Maintains similar length and capitalization.
4. Matches entity-type rules (e.g. person→similar name, location→similar scale).
Output MUST be exactly one JSON object, one line, no extra keys, no code fences:
{"entity": "original entity", "perturbed": "perturbed entity"}
"""


def task_prompt(row, task, defense=False, qa_type="open"):
    text = row["perturbed_text"]
    if task == "summary":
        system, prompt = SUMMARY_SYSTEM, f"Summarize: {text}"
    elif task == "rephrase":
        system, prompt = REPHRASE_SYSTEM, f"Rephrase: {text}"
    elif task == "qa":
        key = "open_question" if qa_type == "open" else "close_question"
        question = row.get("perturbed_" + key) or row.get(key) or row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"QA input needs {key}; run qa_generator.py first")
        system = OPEN_QA_SYSTEM if qa_type == "open" else CLOSED_QA_SYSTEM
        prompt = f"Context: {text[:3000]}\n\nQuestion: {question}\n"
        if qa_type == "closed":
            prompt += f"A: {row['choice_A']}\nB: {row['choice_B']}\n"
    else:
        raise ValueError(f"Unknown task: {task}")
    if defense:
        system += " " + DEFENSE
    return prompt, system
