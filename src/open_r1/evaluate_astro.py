"""Custom evaluation tasks for LightEval."""

import random

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


# Prompt template from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14
GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

SUPER_GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is the correct option letter. Think step by step before answering.

{Question}

{Options}
""".strip()

ASTROBENCH_MCQ_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


gpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)

super_gpqa_metric = multilingual_extractive_match_metric(   
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)


def gpqa_prompt_fn(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"]
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )

def super_gpqa_prompt_fn(line, task_name: str = None):
    # Get the answer letter (e.g., 'A', 'B', etc.)
    answer_letter = line["answer_letter"]
    # Convert answer letter to index (A->0, B->1, etc.)
    gold_index = ord(answer_letter) - ord('A')
    
    # Generate options string dynamically
    options = []
    letters = [chr(ord('A') + i) for i in range(len(line["options"]))]
    for letter, option in zip(letters, line["options"]):
        options.append(f"{letter}) {option}")
    options_str = "\n".join(options)
    
    query = SUPER_GPQA_QUERY_TEMPLATE.format(
        Question=line["question"],
        Options=options_str
    )
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=letters,  # Dynamic list of available choices
        gold_index=gold_index,
        instruction=query,
    )

def astrobench_mcq_prompt_fn(line, task_name: str = None):
    query = ASTROBENCH_MCQ_QUERY_TEMPLATE.format(
        Question=line["question"],
        A=line["A"],
        B=line["B"],
        C=line["C"],
        D=line["D"]
    )
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=0,
        instruction=query,
    )


gpqa_astro = LightevalTaskConfig(
    name="gpqa:astro",
    suite=["custom"],
    prompt_function=gpqa_prompt_fn,
    hf_repo="Maxwell-Jia/gpqa-astro",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metric=[gpqa_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=1,
)

super_gpqa_astro = LightevalTaskConfig(
    name="super_gpqa:astro",
    suite=["custom"],
    prompt_function=super_gpqa_prompt_fn,
    hf_repo="Maxwell-Jia/SuperGPQA-Astro",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metric=[super_gpqa_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=1,
)

astrobench_mcq = LightevalTaskConfig(
    name="astrobench:mcq",
    suite=["custom"],
    prompt_function=astrobench_mcq_prompt_fn,
    hf_repo="AstroMLab/Astrobench_MCQ_v1_Public",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[gpqa_metric],
    stop_sequence=None,
    trust_dataset=True,
    version=1,
)

# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(gpqa_astro)
TASKS_TABLE.append(super_gpqa_astro)
TASKS_TABLE.append(astrobench_mcq)  # Add the new task

# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
