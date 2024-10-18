# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The utility functions for prompt optimization."""
import csv
import collections
import json
import os
import pickle
import sys
from dassl.engine import build_trainer

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

import clip
import numpy as np
import pandas as pd
from optimization import eval_utils
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import re


def extract_string_in_square_brackets(input_string):
    raw_result = re.findall(r"\[.*?\]", input_string)
    if raw_result:
        return raw_result[0][1:-1]
    else:
        return ""


def parse_tag_content(text, prefix="<TEXT>", suffix="</TEXT>"):
    pattern = f"{prefix}(.*?){suffix}"
    results = re.findall(pattern, text, re.DOTALL)
    return results


def _bucketize_float(num, n_buckets=20):
    return num


def gen_ins_and_score_pairs_substr(
        old_instructions_and_scores,
        old_instruction_score_threshold=0.1,
        max_num_instructions=1000,
        return_str_only=False,
        num_score_buckets=np.inf,
):
    """Generate the string that includes instruction-score pairs."""
    assert num_score_buckets == np.inf or isinstance(num_score_buckets, int)
    old_instructions_and_scores_str = ""
    old_instructions_and_scores = sorted(
        # old_instructions_and_scores, key=lambda x: x[1]  #
        old_instructions_and_scores, key=lambda x: (x[1], -x[2])  # entropy_value  scores
        # old_instructions_and_scores, key=lambda x: -x[2]  #
    )[-max_num_instructions:]
    old_instructions_and_scores_in_meta_prompt = []
    for instruction, score, entropy_value, i_step in old_instructions_and_scores:
        if (
                not old_instruction_score_threshold
                or score >= old_instruction_score_threshold
        ):
            old_instructions_and_scores_in_meta_prompt.append(
                (instruction, score, i_step)
            )
            if num_score_buckets == np.inf:
                score_to_show = round(score, 3)
            else:
                score_to_show = _bucketize_float(score, num_score_buckets)
            old_instructions_and_scores_str += (
                f"\ntext:\n{instruction}\nscore:\n{score_to_show}\nloss:\n{entropy_value}\n"  # loss + score
            )
    if return_str_only:
        return old_instructions_and_scores_str
    else:
        return (
            old_instructions_and_scores_str,
            old_instructions_and_scores_in_meta_prompt,
        )


def gen_meta_prompt(
        old_instructions_and_scores,
        instruction_pos,
        optimizer_llm_name,
        old_instruction_score_threshold=0.1,
        max_num_instructions=1000,
        meta_prompt_type="both_instructions_and_exemplars",
        few_shot_qa_pairs=False,
        few_shot_index_list=None,
        instructions_before_exemplars=True,
        num_score_buckets=np.inf,
        dataset_name="",
        task_name="",
        descrip_str="",
):
    """Generate meta prompt for instruction rewriting.

    Args:
     old_instructions_and_scores (list): a list of (instruction, score, i_step)
       pairs.
     instruction_pos (str): where to put the instruction, one of {'before_QA',
       'Q_beginning', 'Q_end', 'A_beginning'}.
     optimizer_llm_name (str): the name of the LLM used for instruction editing.
     old_instruction_score_threshold (float): only add old instructions with score
       no less than this threshold.
     max_num_instructions (int): the maximum number of instructions in the meta
       prompt.
     meta_prompt_type (str): the type of meta-prompt: whether to have both
       previous instructions and dataset exemplars (often for fine-tuned
       optimizers), or to have only previous instructions (often for pre-trained
       optimizers).
     few_shot_qa_pairs (bool): whether to have few-shot QA pairs in the meta
       prompt.
     include_qa (bool): whether to include "Q:" and "A:" formats in the prompt.
     data (list or pd.DataFrame): the raw data.
     few_shot_index_list (list): the list of indices of few-shot examples.
     instructions_before_exemplars (bool): whether the instruction-score pairs are
       before the exemplars from the dataset.
     num_score_buckets (np.inf or int): the number of score buckets when we
       convert float accuracies to integers. Default to np.inf for not
       bucketizing.
     dataset_name (str): the name of the current dataset. Only used when
       generating task description when meta_prompt_type == "instructions_only".
     task_name (str): the name of the current task. Only used when generating task
       description when meta_prompt_type == "instructions_only".

    Returns:
     meta_prompt (str): the generated meta prompt.
    """
    assert instruction_pos in {
        "before_Q",
        "Q_beginning",
        "Q_end",
        "A_beginning",
    }, (
        "The instruction position should be either before the question, or at the"
        " beginning of the question, at the end of the question, or at the"
        " beginning of the answer."
    )
    assert meta_prompt_type in {
        "both_instructions_and_exemplars",
        "instructions_only",
    }
    assert num_score_buckets == np.inf or isinstance(num_score_buckets, int)

    meta_prompt = ""
    if meta_prompt_type == "both_instructions_and_exemplars":
        if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4"}:
            if instruction_pos == "A_beginning":
                meta_prompt_old_instruction_part = (
                    " Your task is to generate the answer starting sentence <Start>."
                    " Below are some previous starting sentences with their scores."
                    " The score ranges from 0 to 100.\n"
                )
            else:
                meta_prompt_old_instruction_part = (
                    "You need to perform image classification on the large scale visual recognition dataset based on visual features.\n"
                    "Here, <CLASS> represents a class name from the large scale visual recognition dataset, and it is essential to include <CLASS> in <INS>.\n"
                    "Below are some previous instructions with their scores and loss. "
                    "The score ranges from 0 to 100. "
                    "The loss ranges from 0 to positive infinity.\n"
                    "\nHere is a description of some features of the flowers in the image:\n"
                )
        else:
            assert optimizer_llm_name.lower() == "text-bison"
            meta_prompt_old_instruction_part = (
                "I have some texts along with their corresponding scores."
                " The texts are arranged in ascending order based on their scores,"
                " where higher scores indicate better quality.\n\n"
            )
        #
        old_instructions_and_scores_str = gen_ins_and_score_pairs_substr(
            old_instructions_and_scores=old_instructions_and_scores,  #
            old_instruction_score_threshold=old_instruction_score_threshold,  #
            max_num_instructions=max_num_instructions,  #
            return_str_only=True,
            num_score_buckets=num_score_buckets  #
        )
        meta_prompt_old_instruction_part = meta_prompt_old_instruction_part + descrip_str
        meta_prompt_old_instruction_part = meta_prompt_old_instruction_part
        meta_prompt_old_instruction_part += old_instructions_and_scores_str

        meta_prompt += meta_prompt_old_instruction_part  #
        if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4"}:
            if instruction_pos == "A_beginning":
                meta_prompt += (
                    "\n\nGenerate a starting sentence that is different from all the"
                    " <Start> sentences above, and has a higher score than all the"
                    " <Start> sentences above. The starting sentence should begin with"
                    " <Start> and end with </Start>. The starting sentence should be"
                    " concise, effective, and generally applicable to all QA pairs"
                    " above."
                )
            else:  #
                meta_prompt += (  #
                    "\n\nGenerate an instruction that differs from all the instructions <INS>　above, with a higher score and a lower loss."
                    " The instruction should begin with <INS> and end with </INS>."
                    " The instruction should be concise, effective,"
                    " and generally applicable to all problems above.\n"
                )



        else:
            assert optimizer_llm_name.lower() == "text-bison"
            meta_prompt += (
                "\n\nWrite your new text that is different from the old ones and"
                " has a score as high as possible. Write the text in square brackets."
            )
    else:
        # when using a pre-trained model as optimizer
        assert meta_prompt_type == "instructions_only"

        assert instruction_pos in {"Q_beginning", "Q_end", "A_beginning"}
        if instruction_pos == "Q_beginning":
            instruction_pos_description = "at the beginning of the question"
        elif instruction_pos == "Q_end":
            instruction_pos_description = "at the end of the question"
        else:
            assert instruction_pos == "A_beginning"
            instruction_pos_description = "at the beginning of the answer"

        if dataset_name == "gsm8k":
            instruction_task_description = "grade school math"
        elif dataset_name == "mmlu":
            instruction_task_description = task_name
        else:
            assert dataset_name == "bbh"
            instruction_task_description = " ".join(task_name.split("_"))

        meta_instruction = (
            f"Create a piece of text {instruction_pos_description.strip()} to"
            " enhance the precision in solving diverse"
            f" {instruction_task_description.strip()} problems."
        )
        old_instructions_and_scores = sorted(
            old_instructions_and_scores, key=lambda x: x[1]
        )
        old_instructions_and_scores_str = ""
        for instruction, score, _ in old_instructions_and_scores:
            if num_score_buckets == np.inf:
                score_to_show = round(score, 2)
            else:
                score_to_show = _bucketize_float(score, num_score_buckets)
            old_instructions_and_scores_str += (
                f"\n\nPrecision: {score_to_show} <TEXT>{instruction}</TEXT>"
            )
        meta_prompt += meta_instruction + old_instructions_and_scores_str
    return meta_prompt


def run_evolution(**kwargs):
    """The function for evolution."""
    # ================= experiment configurations =============================
    num_search_steps = kwargs["num_search_steps"]
    old_instruction_score_threshold = kwargs["old_instruction_score_threshold"]
    optimizer_llm_dict = kwargs["optimizer_llm_dict"]
    tasks_all = kwargs["tasks_all"]
    dataset_name = kwargs["dataset_name"]
    task_name = kwargs["task_name"]
    optimizer_llm_temperature = kwargs["optimizer_llm_temperature"]
    optimizer_llm_temperature_schedule = (
        kwargs["optimizer_llm_temperature_schedule"]
        if "optimizer_llm_temperature_schedule" in kwargs
        else "constant"
    )
    optimizer_llm_temperature_end = (
        kwargs["optimizer_llm_temperature_end"]
        if "optimizer_llm_temperature_end" in kwargs
        else None
    )
    initial_instructions = kwargs["initial_instructions"]
    call_optimizer_server_func = kwargs["call_optimizer_server_func"]
    instruction_pos = kwargs["instruction_pos"]
    prediction_treat_as_number = kwargs["prediction_treat_as_number"]
    prediction_treat_as_bool = kwargs["prediction_treat_as_bool"]
    result_by_instruction_folder = kwargs["result_by_instruction_folder"]
    few_shot_qa_pairs = kwargs["few_shot_qa_pairs"]
    num_score_buckets = kwargs["num_score_buckets"]
    max_num_instructions = kwargs["max_num_instructions"]
    meta_prompt_type = kwargs["meta_prompt_type"]
    meta_prompt_instructions_before_exemplars = kwargs[
        "meta_prompt_instructions_before_exemplars"
    ]
    optimizer_llm_name = kwargs["optimizer_llm_name"]
    num_generated_instructions_in_each_step = kwargs[
        "num_generated_instructions_in_each_step"
    ]
    save_folder = kwargs["save_folder"]
    cfg = kwargs["cfg"]
    # =================== save configurations to json file ====================
    configs_dict = dict()
    configs_dict["optimizer_llm_dict"] = optimizer_llm_dict
    configs_dict["instruction_pos"] = instruction_pos
    configs_dict["optimizer_llm_temperature"] = optimizer_llm_temperature
    configs_dict["optimizer_llm_temperature_schedule"] = (
        optimizer_llm_temperature_schedule
    )
    configs_dict["optimizer_llm_temperature_end"] = optimizer_llm_temperature_end
    with open(os.path.join(save_folder, "configs_dict.json"), "w") as f:
        json.dump(configs_dict, f, indent=4)

    generated_ins_on_few_shot_results_dict = dict()
    old_ins_on_few_shot_results_dict = dict()
    eval_results = []
    old_instructions_and_scores_raw = []
    old_instructions_and_scores = []
    meta_prompts = []  # format: [(meta_prompt, step_index)]
    instruction_score_dict = dict()  # the dictionary of {instruction: score}
    few_shot_index_list_by_step_dict = dict()
    detailed_results_df_by_instruction_dict = dict()
    wrong_questions_from_start_counter = collections.Counter()
    # EVAL results
    eval_detailed_results_df_dict = dict()  # {instruction: detailed_results_df}
    instruction_eval_score_dict = dict()  # {instruction: eval_score}
    old_instruction_md5_hashstrings_set = set()

    print(f"tasks_all: {tasks_all}")
    print(
        f"optimizer llm temperature: {optimizer_llm_temperature}, schedule:"
        f" {optimizer_llm_temperature_schedule}"
    )
    print(
        f"generating {num_generated_instructions_in_each_step} instructions in"
        f" each step, run for {num_search_steps} steps"
    )
    print(
        "discarding generated instructions with score less than:"
        f" {old_instruction_score_threshold} (old_instruction_score_threshold)"
    )
    print(f"num_score_buckets: {num_score_buckets}")

    prev_saved_instructions = set()

    start_batch_idx = 0
    end_batch_idx = 0

    descrip_str = ''
    # evaluate initial instructions
    print("\n============== evaluating initial instructions ===============")
    for instruction in initial_instructions:
        print(f"""computing the score of "{instruction}" by prompting""")

        trainer = build_trainer(cfg)

        acc, entropy_value = trainer.train_llm(start_batch_idx, end_batch_idx, instruction)  #
        # Textual descriptions of training images
        with open('/desc/oxford_flowers_1_shot_image.txt', 'r', encoding='utf-8') as file:
            descrip = file.readlines()

        descrip_str = '\n'.join(descrip)
        average_score = acc

        print(f"instruction: {instruction}, score: {average_score}, loss: {entropy_value}")  # loss + score
        filename = eval_utils.instruction_to_filename(instruction)

        # Create the full file path
        file_path = os.path.join(result_by_instruction_folder, f"{filename}.txt")
        # Open the file in write mode and create a CSV writer
        with open(file_path, 'w') as file:
            file.write(f"instruction\n: {instruction}\n")
            file.write(f"score\n: {average_score}\n")
            file.write(f"loss\n: {entropy_value}\n")

        old_instructions_and_scores.append((instruction, average_score, entropy_value, -1))
        old_instructions_and_scores_raw.append((instruction, average_score, entropy_value, -1))
        instruction_score_dict[instruction] = average_score

    # evolution
    for i_step in range(num_search_steps):
        print(f"\n================== Step {i_step} =====================")
        if not i_step % 10:
            print(f"old_instructions_and_scores: {old_instructions_and_scores}")

        if optimizer_llm_temperature_schedule == "linear_increase":
            optimizer_llm_temperature_curr = (
                    optimizer_llm_temperature
                    + i_step
                    / num_search_steps
                    * (optimizer_llm_temperature_end - optimizer_llm_temperature)
            )
        else:
            optimizer_llm_temperature_curr = optimizer_llm_temperature
        # print(f"current optimizer_llm_temperature: {optimizer_llm_temperature_curr}")

        # generate new instructions
        meta_prompt = gen_meta_prompt(
            old_instructions_and_scores=old_instructions_and_scores,
            instruction_pos=instruction_pos,
            optimizer_llm_name=optimizer_llm_name,
            old_instruction_score_threshold=old_instruction_score_threshold,
            max_num_instructions=max_num_instructions,
            meta_prompt_type=meta_prompt_type,
            few_shot_qa_pairs=False,
            instructions_before_exemplars=meta_prompt_instructions_before_exemplars,
            num_score_buckets=num_score_buckets,
            dataset_name=dataset_name,
            task_name=task_name,
            descrip_str=descrip_str
        )
        print(f"\nmeta_prompt: \n\n{meta_prompt}\n")
        meta_prompts.append((meta_prompt, i_step))
        remaining_num_instructions_to_generate = (
            num_generated_instructions_in_each_step  # 8
        )
        generated_instructions_raw = []
        while remaining_num_instructions_to_generate > 0:  #
            optimizer_llm_input_text = meta_prompt
            # generate instructions
            # print(f"current temperature: {optimizer_llm_temperature_curr}")
            raw_outputs = call_optimizer_server_func(
                optimizer_llm_input_text,
                temperature=optimizer_llm_temperature_curr,
            )

            # Extract the generated instructions from the optimizer LLM output. Only
            # keep some samples if the desired number of remaining instructions
            # is smaller than the total number of decodes in this step.
            if meta_prompt_type == "both_instructions_and_exemplars":
                raw_outputs = raw_outputs[:remaining_num_instructions_to_generate]
                if optimizer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4"}:
                    if instruction_pos == "A_beginning":
                        start_string = "<Start>"
                        end_string = "</Start>"
                    else:
                        start_string = "<INS>"
                        end_string = "</INS>"
                    for raw_output in raw_outputs:
                        if start_string not in raw_output:
                            start_index = 0
                        else:
                            start_index = raw_output.index(start_string) + len(start_string)
                        if end_string not in raw_output:
                            end_index = len(raw_output)
                        else:
                            end_index = raw_output.index(end_string)
                        new_inst = raw_output[start_index:end_index].strip()
                        generated_instructions_raw.append(new_inst)
                else:
                    assert optimizer_llm_name.lower() == "text-bison"
                    generated_instructions_raw += [
                        extract_string_in_square_brackets(string)
                        for string in raw_outputs
                    ]

                remaining_num_instructions_to_generate -= optimizer_llm_dict[
                    "batch_size"
                ]
            else:
                assert meta_prompt_type == "instructions_only"
                max_num_instructions_to_keep_in_each_output = 1
                for string in raw_outputs:
                    generated_instructions_raw += parse_tag_content(string)[
                                                  :max_num_instructions_to_keep_in_each_output
                                                  ]
                remaining_num_instructions_to_generate -= (
                        optimizer_llm_dict["batch_size"]
                        * max_num_instructions_to_keep_in_each_output
                )

        generated_instructions_raw = list(
            map(eval_utils.polish_sentence, generated_instructions_raw)
        )
        print(f"\ninitially generated instructions: {generated_instructions_raw}\n")

        # do not evaluate old instructions again
        generated_instructions = []  # the new instructions generated in this step
        for ins in generated_instructions_raw:
            ins_md5_hashstring = eval_utils.instruction_to_filename(
                ins, md5_hashing=True
            )
            if ins_md5_hashstring not in old_instruction_md5_hashstrings_set:
                generated_instructions.append(ins)  #
                old_instruction_md5_hashstrings_set.add(ins_md5_hashstring)  #
            else:
                print(f"already evaluated '{ins}' previously")
        generated_instructions = list(set(generated_instructions))

        to_evaluate_instructions = []
        for instruction in generated_instructions:
            if torch.nonzero(clip.tokenize(instruction)).size(0) > 77:
                print("tokenize:", torch.nonzero(clip.tokenize(instruction)).size(0))
                continue
            if len(instruction) > 500:
                print(f"Step {i_step}, instruction: {instruction}, too long, skipped")
                continue
            if dataset_name == "gsm8k" and any(
                    char.isdigit() for char in instruction
            ):
                print(
                    f"Step {i_step}, instruction: {instruction}, contains numbers,"
                    " skipped"
                )
                continue
            if "INS" in instruction:
                print(
                    f"Step {i_step}, instruction: {instruction}, contains 'INS',"
                    " skipped"
                )
                continue
            to_evaluate_instructions.append(instruction)
        print(f"\nto-evaluate generated instructions: {to_evaluate_instructions}\n")  #

        # evaluate new instructions on the few-shot exemplars in meta-prompt
        # evaluate OLD instructions on the few-shot exemplars in meta-prompt
        # evaluate newly generated instructions on the training set
        for instruction in to_evaluate_instructions:  #
            if instruction not in prev_saved_instructions:  #
                print(f"""computing the score of "{instruction}" by prompting""")

                trainer = build_trainer(cfg)
                start_batch_idx = start_batch_idx + 1
                end_batch_idx = end_batch_idx + 1
                acc, entropy_value = trainer.train_llm(start_batch_idx, end_batch_idx, instruction)  #
                # coop result
                score = acc
                prev_saved_instructions.add(instruction)
            else:
                trainer = build_trainer(cfg)
                start_batch_idx = start_batch_idx + 1
                end_batch_idx = end_batch_idx + 1
                acc, entropy_value = trainer.train_llm(start_batch_idx, end_batch_idx, instruction)
                score = acc
                print(f"""reading previously saved "{instruction}" information""")


            scores = score
            print(
                f"Step {i_step}, instruction: {instruction}, score: {scores},loss: {entropy_value}"
            )

            filename = eval_utils.instruction_to_filename(instruction)

            file_path = os.path.join(result_by_instruction_folder, f"{filename}.txt")
            # Create the full file path
            # Open the file in write mode and create a CSV writer
            with open(file_path, 'w') as file:
                file.write(f"i_step\n: {i_step}\n")
                file.write(f"instruction\n: {instruction}\n")
                file.write(f"score\n: {scores}\n")

            # print(f"saving results to {file_path}")
            # Create the full file path
            # Open the file in write mode and create a CSV writer
            with open(file_path, 'w') as file:
                file.write(f"instruction\n: {instruction}\n")
                file.write(f"score\n: {average_score}\n")
                file.write(f"loss\n: {entropy_value}\n")



            old_instructions_and_scores.append((instruction, scores, entropy_value, i_step))
            instruction_score_dict[instruction] = scores

        # record all generated instructions
        for instruction in generated_instructions_raw:
            if instruction in instruction_score_dict:
                average_score = instruction_score_dict[instruction]
            else:
                average_score = np.nan
            old_instructions_and_scores_raw.append(
                (instruction, average_score, i_step)
            )

        # ===================== save up-to-date results ===========================
        results_dict = dict()
        results_dict["meta_prompts"] = meta_prompts
        results_dict["old_instructions_and_scores"] = list(
            old_instructions_and_scores
        )
        results_dict["old_instructions_and_scores_raw"] = list(
            old_instructions_and_scores_raw
        )
        results_dict["generated_ins_on_few_shot_results_dict"] = (
            generated_ins_on_few_shot_results_dict
        )
        results_dict["old_ins_on_few_shot_results_dict"] = (
            old_ins_on_few_shot_results_dict
        )
        results_dict["few_shot_index_list_by_step_dict"] = (
            few_shot_index_list_by_step_dict
        )
        results_dict["eval_results"] = eval_results
        results_dict["eval_detailed_results_df_dict"] = (
            eval_detailed_results_df_dict
        )
        with open(os.path.join(save_folder, "results_dict.pkl"), "wb") as fp:
            pickle.dump(results_dict, fp)
        # print(f"\nsaved all results to\n{save_folder}")
