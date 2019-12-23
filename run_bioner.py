# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

isdebugging = False

import csv
import os
import logging
import argparse
import random
import datetime
import subprocess
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss

import tokenization
from modeling import BertConfig, BertForNER
from optimization import BERTAdam
from tensorboardX import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_a_map=None, text_b_map=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.text_a_map = text_a_map
        self.text_b_map = text_b_map


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
        

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class RjnlpbaProcessor(DataProcessor):
    """Processor for the Revised JNLPBA data set (IASL Academia Sinica version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_multiline(os.path.join(data_dir, "Genia4ERtask1.iob2")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_multiline(os.path.join(data_dir, "Genia4EReval1.iob2")), "dev")

    def get_labels(self):
        """See base class."""
        labels = ["[PAD]", "O"]
        for pref in ["B", "I"]:
            for netype in ["protein", "DNA", "RNA", "cell_line", "cell_type"]:
                labels.append(pref + '-' + netype)
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples
    
    def _read_multiline(self, datafile):
        sentences_and_labels = []
        sent = []
        labs = []
        with open(datafile) as dfile:
            line_num = 0
            for ll in dfile.readlines():
                line_num += 1
                # ------------debugging
                if isdebugging and (line_num > 1000):
                    break
                # ------------debugging
                ll = ll.strip()
                if len(ll) < 1:
                    if len(sent) > 0:
                        assert len(sent) == len(labs), "Sent len diff from labs"
                        # sent = ' '.join(sent)
                        sentences_and_labels.append((sent, labs))
                        sent = []
                        labs = []
                        continue
                l_split = ll.split('\t')
                if len(l_split) != 2:
                    print("Line {} error".format(line_num))
                word, netag = l_split
                sent.append(word)
                labs.append(netag)
            if len(sent) > 0: # last sentence
                assert len(sent) == len(labs), "Sent len diff from labs"
                # sent = ' '.join(sent)
                sentences_and_labels.append((sent, labs))
                sent = []
                labs = []
        return sentences_and_labels

class AICupProcessor(DataProcessor):
    """Processor for the AICup data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_multiline(os.path.join(data_dir, "training/train_conll.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            # self._read_multiline("sho.tsv"), "dev")
            self._read_multiline(os.path.join(data_dir, "development/test_only_in_list_conll.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_multiline(os.path.join(data_dir, "test/test_only_in_list_conll.tsv")), "test")
            # self._read_multiline(os.path.join(data_dir, "test/test_conll.tsv")), "test")

    def get_labels(self):
        """See base class."""
        labels = ["[PAD]", "O", "[CLS]", "[SEP]"]
        for pref in ["B", "I"]:
            for netype in ["Gene", "Chemical", "Disease", "Partial_Gene"]:
                labels.append(pref + '-' + netype)
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples
    
    def _read_multiline(self, datafile):
        sentences_and_labels = []
        sent = []
        labs = []
        with open(datafile) as dfile:
            line_num = 0
            for ll in dfile.readlines():
                line_num += 1
                # ------------debugging
                if isdebugging and (line_num > 1000):
                    break
                # ------------debugging
                ll = ll.strip()
                if len(ll) < 1:
                    if len(sent) > 0:
                        assert len(sent) == len(labs), "Sent len diff from labs"
                        # sent = ' '.join(sent)
                        sentences_and_labels.append((sent, labs))
                        sent = []
                        labs = []
                        continue
                l_split = ll.split('\t')
                if len(l_split) != 2:
                    print("Line {} error".format(line_num))
                word, netag = l_split
                sent.append(word)
                labs.append(netag)
            if len(sent) > 0: # last sentence
                assert len(sent) == len(labs), "Sent len diff from labs"
                # sent = ' '.join(sent)
                sentences_and_labels.append((sent, labs))
                sent = []
                labs = []
        return sentences_and_labels

def add_all_arguments(parser_object):
    ## Required parameters
    parser_object.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser_object.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser_object.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser_object.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser_object.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser_object.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser_object.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser_object.add_argument("--max_seq_length",
                        default=100,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser_object.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser_object.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser_object.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser_object.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser_object.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser_object.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser_object.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser_object.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser_object.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser_object.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser_object.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser_object.add_argument('--seed', 
                        type=int, 
                        default=1337,
                        help="random seed for initialization")
    parser_object.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser_object.add_argument("--less_log",
                        default=False,
                        action='store_true',
                        help="Show only warnings")
    
    return parser_object    

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = dict()
    for label in label_list:
        label_map[label] = len(label_map)

    reverse_label_map = dict([(v,k) for (k,v) in label_map.items()])
    max_len_in_data = 0

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens_a, tokens_map_a = tokenizer.tokenize_with_map(example.text_a)
        example.text_a_map = tokens_map_a
        if len(tokens_a) > max_len_in_data:
            max_len_in_data = len(tokens_a)
            max_len_in_data_tokens = tokens_a
        tokens_b = None
        if example.text_b:
            tokens_b, tokens_map_b = tokenizer.tokenize_with_map(example.text_b)
            example.text_b_map = tokens_map_b
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]
            # Account for [SEP] with "-1"
            # if len(tokens_a) > max_seq_length - 1:
            #     tokens_a = tokens_a[0:(max_seq_length - 1)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        # try to not add [CLS]
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        label_ids = [label_map["[CLS]"]] # `[CLS]' symbol
        # label_ids = [] # no `[CLS]' symbol
        
        for ori_pos in tokens_map_a:
            t_l = example.label[ori_pos]
            label_ids.append(label_map[t_l])
            if len(label_ids) == len(input_ids) - 1: # exclude last [SEP]
                break

        label_ids.append(label_map["[SEP]"]) # `[SEP]' symbol

        assert len(label_ids) == len(input_ids), "Label and sent len diff: {} and {}".format(len(label_ids), len(input_ids))

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("labels: %s (id = %s)" % (' '.join([reverse_label_map[ll] for ll in label_ids]), label_ids))
        
        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_ids))
    logger.info("Max length in data {}: {}".format(max_len_in_data, ' '.join(max_len_in_data_tokens)))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def save_model(model, args, suffix=''):
    if len(suffix) > 1:
        suffix = '_' + suffix
    with open('{}/model{}.pt'.format(args.output_dir, suffix), 'wb') as f:
        torch.save(model.state_dict(), f)

def create_data_loader(features, args, set_name):
    global logger
    assert set_name in ["train","eval","test"], "Set name error {}".format(set_name)
        
    batch_size = args.train_batch_size if set_name == "train" else args.eval_batch_size

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    t_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        if set_name == "train":
            logger.info("Randomizing training set")
            t_sampler = RandomSampler(t_data)
        else:
            t_sampler = SequentialSampler(t_data)
    else:
        t_sampler = DistributedSampler(t_data)
    t_dataloader = DataLoader(t_data, sampler=t_sampler, batch_size=batch_size)
    return t_dataloader

def get_model_predictions(model, data_iterator, label_id_to_name, device):
    _accuracy = []
    _predictions = []
        
    for input_ids, input_mask, segment_ids, label_ids in data_iterator:

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        label_ids = label_ids.view(-1)
        
        logits = model(input_ids, segment_ids, input_mask, label_ids)
        pred_labels = logits.max(-1)[1]
        for _, p in enumerate(pred_labels):
            p_l = [label_id_to_name[l] for l in p.cpu().numpy().tolist()]
            _predictions.append(p_l)
                
        pred_labels = pred_labels.view(-1)
            
        not_pads = torch.gt(label_ids, 0).float()
        corrects = torch.eq(pred_labels, label_ids).float()
        corrects = torch.sum(torch.mul(corrects, not_pads))
        
        not_pads = torch.sum(not_pads)
        tmp_accuracy = 0.
        if not_pads > 1e-8:
            tmp_accuracy = (corrects / not_pads).item()
        _accuracy.append(tmp_accuracy)
    
    return _predictions, _accuracy

def write_predictions(predictions, examples, filename):
    with open(filename, "w") as writer:
        for i_sent, sent_example in enumerate(examples):
            preds = predictions[i_sent][1:] # exclude [CLS]
            sent_toks = sent_example.text_a
            sent_toks_map = sent_example.text_a_map

            restored_tags = []
            for i_prd, pred_ne_tag in enumerate(preds):
                if i_prd >= len(sent_toks_map):
                    break
                if (i_prd > 0) and (sent_toks_map[i_prd - 1] == sent_toks_map[i_prd]):
                    continue
                restored_tags.append(pred_ne_tag)

            while len(restored_tags) < len(sent_toks):
                restored_tags.append('O')
            assert len(restored_tags) == len(sent_toks), \
                   "Restored tags and sentence diff in len: tags: {}, sent: {}".format( \
                                                            len(restored_tags), len(sent_toks), \
                                                                                      )

            _ = [writer.write("{}\t{}\n".format(w,t)) for w,t in zip(sent_toks, restored_tags)]
            writer.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser)
    args = parser.parse_args()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "rjnlpba": RjnlpbaProcessor,
        "aicup": AICupProcessor,
    }

    if args.less_log:
        logger.setLevel(logging.WARNING)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    # args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    eval_examples = None
    num_train_steps = None
    
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
    
    model = BertForNER(bert_config, len(label_list))

    if args.init_checkpoint is not None:
        model_params_dict = model.state_dict()
        pretrained_dict = torch.load(args.init_checkpoint, map_location='cpu')
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_params_dict}
        # 2. overwrite entries in the existing state dict
        model_params_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_params_dict)

    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = model.named_parameters()
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0

    if args.do_train:
        summary_dir_name = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        log_dir_name = os.path.join(args.output_dir, "tbx", summary_dir_name)
        train_summary_writer = SummaryWriter(log_dir=log_dir_name)

        loss_weights = [1.] * len(label_list)
        loss_weights[0] = 0.01
        # loss_weights[1] = 0.1
        loss_weights = torch.tensor(loss_weights).float().to(device)
        loss_func = CrossEntropyLoss(ignore_index=0, weight=loss_weights)
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        train_dataloader = create_data_loader(train_features, args, "train")
        if args.do_eval:
            eval_dataloader = create_data_loader(eval_features, args, "eval")

        for _ in trange(args.num_train_epochs, desc="Epoch"):
            iter_tqdm = tqdm(train_dataloader, desc="Iter")
            for step, batch in enumerate(iter_tqdm):
                torch.cuda.empty_cache()
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                label_ids = label_ids.view(-1)
                logits = model(input_ids, segment_ids, input_mask, label_ids)
                logits = logits.view((-1, logits.size()[-1]))
                loss = loss_func(logits, label_ids)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                training_loss = loss.item()
                train_summary_writer.add_scalar('training/loss', training_loss, global_step)
                iter_tqdm.set_postfix_str("loss {:.4f}".format(loss.item()))

                pred_labels = logits.max(-1)[1]
                not_pads = torch.gt(label_ids, 1).float()
                corrects = torch.eq(pred_labels, label_ids).float()
                train_accuracy = torch.sum(torch.mul(corrects, not_pads)) / torch.sum(not_pads)
                train_summary_writer.add_scalar('training/accuracy', train_accuracy, global_step)
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1
                
                    if global_step % args.save_checkpoints_steps == 0:
                        save_model(model, args, suffix='step_{}'.format(global_step))
                        # show some sample outputs
                        # logits = logits.detach().cpu()[:10]
                        # tmp_mod_out_labs = logits.max(-1)[1]
                        # tmp_mod_out_labs = [label_list[int(ll)] for ll in tmp_mod_out_labs]
                        # tmp_label_ids = label_ids[:10]
                        # tmp_label_ids = [label_list[int(ll)] for ll in tmp_label_ids]
                        # tqdm.write("Pred: {}".format(tmp_mod_out_labs))
                        # tqdm.write("Ans: {}".format(tmp_label_ids))
                        # tqdm.write("Saving model step {}".format(global_step + 1))
                        
                        # run eval set
                        if args.do_eval:
                            model.eval()
                            with torch.no_grad():
                                eval_preds, eval_acc = get_model_predictions(model, eval_dataloader, label_list, device)
                            eval_acc = np.mean(eval_acc)
                            train_summary_writer.add_scalar('eval/accuracy', eval_acc, global_step)
                            output_eval_file = os.path.join(args.output_dir, "predictions.txt")
                            answer_file = os.path.join(args.data_dir, "development/test_only_in_list_conll.tsv")
                            write_predictions(eval_preds, eval_examples, output_eval_file)
                            eval_fscore = subprocess.check_output('./get_score.sh {} {}'.format(output_eval_file, answer_file), shell=True)
                            eval_fscore = float(eval_fscore) / 100
                            train_summary_writer.add_scalar('eval/fscore', eval_fscore, global_step)

        # training end, save last model
        logger.info("***** Training end *****")
        logger.info("Saving final model at step {}".format(global_step))
        save_model(model, args, suffix='step_{}'.format(global_step))
        train_summary_writer.close()

    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_dataloader = create_data_loader(eval_features, args, "eval")

        model.eval()
        with torch.no_grad():
            predictions, eval_accuracy = get_model_predictions(model, eval_dataloader, label_list, device)

        eval_accuracy = np.mean(eval_accuracy)

        output_eval_file = os.path.join(args.output_dir, "predictions.txt")
        write_predictions(predictions, eval_examples, output_eval_file)
        answer_file = os.path.join(args.data_dir, "development/test_only_in_list_conll.tsv")
        eval_fscore = subprocess.check_output('./get_score.sh {} {}'.format(output_eval_file, answer_file), shell=True)
        eval_fscore = float(eval_fscore) / 100
        result = {'eval_accuracy': eval_accuracy,
                  'eval_fscore': eval_fscore,
                  }
        logger.info("***** Eval results *****")
        for key in result.keys():
            logger.info("  %s = %s", key, str(result[key]))

    if args.do_test:

        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer)

        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        test_dataloader = create_data_loader(test_features, args, "test")

        model.eval()
        iter_tqdm_test = tqdm(test_dataloader, desc="Batch")
        with torch.no_grad():
            predictions, test_accuracy = get_model_predictions(model, iter_tqdm_test, label_list, device)

        test_accuracy = np.mean(test_accuracy)

        output_test_pred_file = os.path.join(args.output_dir, "predictions_test.txt")
        write_predictions(predictions, test_examples, output_test_pred_file)
        # answer_file = os.path.join(args.data_dir, "test/test_conll.tsv")
        answer_file = os.path.join(args.data_dir, "test/test_only_in_list_conll.tsv")
        test_fscore = 0.
        try:
            test_fscore = subprocess.check_output('./get_score.sh {} {}'.format(output_test_pred_file, answer_file), shell=True)
            test_fscore = float(test_fscore) / 100
        except:
            pass
        result = {'test_accuracy': test_accuracy,
                  'test_fscore': test_fscore,
                  }
        logger.info("***** Test results *****")
        for key in result.keys():
            logger.info("  %s = %s", key, str(result[key]))

if __name__ == "__main__":
    main()
