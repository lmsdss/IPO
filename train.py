import argparse
import torch
# -*- coding: utf-8 -*-

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import torch
import prompt_utils
import argparse
import datetime
import functools
import os
import sys
import argparse

from absl import app
from absl import flags
import google.generativeai as palm
import numpy as np
import openai
from optimization import opt_utils


# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.zsclip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        # print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    openai_api_key = ""
    palm_api_key = ""
    optimizer_llm_name = "gpt-3.5-turbo"  # "gpt-3.5-turbo"　"gpt-4"
    dataset_name = "prompt_learning"
    task_name = "train"
    meta_prompt_type = "both_instructions_and_exemplars"
    instruction_pos = "Q_beginning"  # A_beginning Q_beginning

    # make sure the scorer and optimizer models are callable
    if optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4"}:
        openai.api_key = openai_api_key
    else:
        assert (
            palm_api_key
        ), "A PaLM API key is needed when prompting the text-bison model."
        palm.configure(api_key=palm_api_key)

    OPRO_ROOT_PATH = "/home/IPO"

    # =================== create the result directory ==========================   datetime
    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )

    save_folder = os.path.join(
        OPRO_ROOT_PATH,
        "LLM_outputs",
        "optimization-results",
        f"{dataset_name.upper()}-{task_name}-o-{optimizer_llm_name}-{datetime_str}/",
    )
    result_by_instruction_folder = os.path.join(
        save_folder, "result_by_instruction"
    )
    os.makedirs(result_by_instruction_folder)

    args.gpu = 'cuda'
    set_random_seed(args.seed)
    # print("Use GPU: {} for training".format(args.gpu))

    # ====================== optimizer model configs ============================
    if optimizer_llm_name.lower() == "text-bison":
        # when prompting text-bison with Cloud API
        optimizer_finetuned_palm_temperature = 1.0
        optimizer_finetuned_palm_num_decodes = 8
        optimizer_finetuned_palm_max_decode_steps = 1024
        optimizer_finetuned_palm_batch_size = 1
        optimizer_finetuned_palm_num_servers = 1
        optimizer_finetuned_palm_dict = dict()
        optimizer_finetuned_palm_dict["temperature"] = (
            optimizer_finetuned_palm_temperature
        )
        optimizer_finetuned_palm_dict["num_decodes"] = (
            optimizer_finetuned_palm_num_decodes
        )
        optimizer_finetuned_palm_dict["batch_size"] = (
            optimizer_finetuned_palm_batch_size
        )
        optimizer_finetuned_palm_dict["num_servers"] = (
            optimizer_finetuned_palm_num_servers
        )
        optimizer_finetuned_palm_dict["max_decode_steps"] = (
            optimizer_finetuned_palm_max_decode_steps
        )

        call_optimizer_finetuned_palm_server_func = functools.partial(
            prompt_utils.call_palm_server_from_cloud,
            model="text-bison-001",
            temperature=optimizer_finetuned_palm_dict["temperature"],
            max_decode_steps=optimizer_finetuned_palm_dict["max_decode_steps"],
        )

        optimizer_llm_dict = {
            "model_type": optimizer_llm_name.lower(),
        }
        optimizer_llm_dict.update(optimizer_finetuned_palm_dict)
        call_optimizer_server_func = call_optimizer_finetuned_palm_server_func

    else:
        assert optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4"}
        optimizer_gpt_max_decode_steps = 40  # 512
        optimizer_gpt_temperature = 1.0  # 1.0

        optimizer_llm_dict = dict()
        optimizer_llm_dict["max_decode_steps"] = optimizer_gpt_max_decode_steps
        optimizer_llm_dict["temperature"] = optimizer_gpt_temperature
        optimizer_llm_dict["batch_size"] = 1
        optimizer_llm_dict["num_decodes"] = 1
        call_optimizer_server_func = functools.partial(
            prompt_utils.call_openai_server_func,
            model=optimizer_llm_name,
            max_decode_steps=optimizer_gpt_max_decode_steps,
            temperature=optimizer_gpt_temperature,
        )

    # print("\n================ prompt optimization settings ==============")
    tasks_all = [task_name]
    prediction_treat_as_number = True
    prediction_treat_as_bool = False

    # ========== set other optimization experiment hyperparameters ==============
    old_instruction_score_threshold = 20  # 0.3
    optimizer_llm_temperature = optimizer_llm_dict["temperature"]
    initial_instructions = [
        "a photo of a <CLASS>.",
    ]

    num_generated_instructions_in_each_step = 8  # 8  #
    num_search_steps = 100  #
    max_num_instructions = 30  # 20 the maximum number of instructions and scores in the meta-prompt

    few_shot_qa_pairs = False  # coop: no qa pairs

    # The number of buckets when converting scores to integers in the meta-prompt.
    num_score_buckets = 100  #
    # whether to put old instructions and scores to before exemplars in
    # the meta-prompt
    meta_prompt_instructions_before_exemplars = True  #

    if args.test:

        trainer = build_trainer(cfg)

        if args.eval_only:
            trainer.load_model(args.model_dir, epoch=args.load_epoch)
            trainer.test()
            return

        if not args.no_train:
            trainer.train()  #
    else:
        evolution_kwargs = {
            "num_search_steps": num_search_steps,
            "old_instruction_score_threshold": old_instruction_score_threshold,
            "optimizer_llm_dict": optimizer_llm_dict,
            "tasks_all": tasks_all,
            "dataset_name": dataset_name,
            "task_name": task_name,
            "optimizer_llm_temperature": optimizer_llm_temperature,
            "initial_instructions": initial_instructions,
            "call_optimizer_server_func": call_optimizer_server_func,
            "instruction_pos": instruction_pos,
            "prediction_treat_as_number": prediction_treat_as_number,
            "prediction_treat_as_bool": prediction_treat_as_bool,
            "result_by_instruction_folder": result_by_instruction_folder,
            "few_shot_qa_pairs": few_shot_qa_pairs,
            "num_score_buckets": num_score_buckets,
            "max_num_instructions": max_num_instructions,
            "meta_prompt_type": meta_prompt_type,
            "meta_prompt_instructions_before_exemplars": (
                meta_prompt_instructions_before_exemplars
            ),
            "optimizer_llm_name": optimizer_llm_name,
            "num_generated_instructions_in_each_step": (
                num_generated_instructions_in_each_step
            ),
            "save_folder": save_folder,
            "cfg": cfg

        }
        opt_utils.run_evolution(**evolution_kwargs)
        #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/data", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="output/base2new/train_base/oxford_flowers/shots_1/IPO/vit_b16_ctxv1/seed1",
                        help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/IPO/vit_b16_ctxv1.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/oxford_flowers.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="CoOp", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    parser.add_argument("--test", type=str, default=False, help="train or test")  #

    args = parser.parse_args()
    main(args)
