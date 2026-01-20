from __future__ import annotations

from typing import Optional


DEFAULT_SYSTEM_MESSAGE: str = "You are a helpful assistant."

SUMMARY_PROMPT: str = "Summarize the given text:"
REPHRASE_PROMPT: str = "Rephrase the given text:"

DEFENSE_PROMPT: str = (
    "Only use the context and knowledge in the given text. "
    "DO NOT use the interior knowledge."
)

DEFENSE_PROMPT_V2: str = (
    "Only use the context and knowledge in the given text. "
    "DO NOT use the interior knowledge.\n"
    "Example:\n"
    "Original entity in context: Albert Einstein\n"
    "Entity in interior knowledge: Isaac Newton\n"
    "Please be loyal to original entity in context."
)

OPEN_QA_GEN_TEMPLATE: str = (
    "Based on the following text, create an open-ended question that has the answer: {answer}\n\n"
    "Text: {text}\n\n"
    "Return format should only include the question itself, without any explanations or prefixes."
)

CLOSE_QA_GEN_TEMPLATE: str = (
    "Based on the following text, create a question with two options A and B.\n\n"
    "Text: {text}\n\n"
    "Option A should be: {original}\n"
    "Option B should be: {perturbed}\n\n"
    "Design a question that can be answered using these two options. The question should relate "
    "to the text content but should not directly indicate which option is the correct answer. \n"
    "Only return the question content, do not include the options.\n\n"
    "Return format should be:\n"
    "Question content?"
)

OPEN_QA_ANSWER_PROMPT: str = (
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Please answer with just the entity name, no explanations."
)

CLOSE_QA_ANSWER_PROMPT: str = (
    "Context: {context}\n\n"
    "Question: {question}\n"
    "A: {choice_a}\n"
    "B: {choice_b}\n\n"
    "Please answer with just the letter of the correct option (A or B), no explanations."
)


def build_task_prompt(task: str, text: str, defense: Optional[str] = None) -> str:
    if task == "summary":
        if defense == DEFENSE_PROMPT:
            return (
                "Summarize the given text. "
                "Only use the context and knowledge in the given text, DO NOT use the interior knowledge.\n\n"
                f"Text: {text}"
            )
        if defense == DEFENSE_PROMPT_V2:
            return (
                "Summarize the given text. "
                "Only use the context and knowledge in the given text, DO NOT use the interior knowledge.\n"
                "Example:\n"
                "Original entity in context: Albert Einstein\n"
                "Entity in interior knowledge: Isaac Newton\n"
                "Please be loyal to original entity in context.\n\n"
                f"Text: {text}"
            )
        return f"Summarize the given text: {text}"
    if task == "rephrase":
        if defense == DEFENSE_PROMPT:
            return (
                "Rephrase the given text. "
                "Only use the context and knowledge in the given text, DO NOT use the interior knowledge.\n\n"
                f"Text: {text}"
            )
        return f"Rephrase the given text: {text}"
    raise ValueError("task must be 'summary' or 'rephrase'")
