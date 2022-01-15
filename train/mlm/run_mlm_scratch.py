# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


import logging
import math
import os
import random
import torch

from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

from my_tokenizer import MyTokenizer


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    vocab: str = field(
        default='mydata/vocab.txt', metadata={"help": "path of vocab"}
    )
    mask_count: int = field(
        default=2,
        metadata={"help": "The maximum number of masked words."}
    )


class MyDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, mask_count: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)
        block_size -= tokenizer.num_special_tokens_to_add(pair=False)

        self.examples = []
        self.mask_starts = []
        self.mask_ends = []
        n_discard = 0
        fin = open(file_path, 'r', encoding='utf-8')
        for __s in fin:
            s = __s.strip()
            if len(s) <= 0:
                continue
            tokens = list(s.replace(' ', ''))
            if len(tokens) > block_size:
                tokens = tokens[ : block_size]
            __input_tokens_id = tokenizer.convert_tokens_to_ids(tokens)
            input_tokens_id = tokenizer.build_inputs_with_special_tokens(__input_tokens_id)
            self.examples.append(input_tokens_id)
            
            L = len(tokens)
            words = s.split(' ')
            mask_start = []
            mask_end = []
            start = 1
            for w in words:
                end = min(start + len(w), L + 1)
                if end - start <= mask_count:
                    mask_start.append(start)
                    mask_end.append(end)
                else:
                    for i in range(start, end - mask_count + 1):
                        mask_start.append(i)
                        mask_end.append(i + mask_count)
                if end > L:
                    break
                start = end
            self.mask_starts.append(mask_start)
            self.mask_ends.append(mask_end)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long), self.mask_starts[i], self.mask_ends[i]


@dataclass
class MyDataCollator:
    tokenizer: PreTrainedTokenizer

    def __call__(
        self, examples: List[Tuple[torch.Tensor, List[int], List[int]]]
    ) -> Dict[str, torch.Tensor]:
        token_ids = [e[0] for e in examples]
        batch = self._tensorize_batch(token_ids)
        starts = []
        ends = []
        for e in examples:
            x = random.randint(0, len(e[1]) - 1)
            starts.append(e[1][x])
            ends.append(e[2][x])
        inputs, labels = self.mask_tokens(batch, starts, ends)
        return {"input_ids": inputs, "labels": labels}

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.Tensor(e) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor, starts: List[int], ends: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        masked_indices = torch.zeros(labels.shape).byte()
        for i, (start, end) in enumerate(zip(starts, ends)):
            masked_indices[i][start : end] = 1

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).byte() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).byte() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return MyDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, mask_count=args.mask_count)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer = MyTokenizer(data_args.vocab)
    config.hidden_size = 512
    config.intermediate_size = 2048
    config.max_position_embedding = 384
    config.num_attention_heads = 8
    config.num_hidden_layers = 6
    config.pooler_fc_size = 512
    config.pooler_num_attention_heads = 8
    config.pooler_size_per_head = 64
    logger.info('Modified config: %s' % str(config))

    '''model = AutoModelWithLMHead.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )'''

    model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    logger.info('Resized vocab: %d' % len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_collator = MyDataCollator(tokenizer=tokenizer)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=not training_args.do_predict,
        tokenizer=tokenizer
    )

    # Training
    if training_args.do_train:
        trainer.train()

    if training_args.do_predict:
        eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        print(trainer.evaluate(eval_dataset))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
