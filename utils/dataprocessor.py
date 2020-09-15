# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Data preprocessing for transformer."""

import io
import time
import logging
import numpy as np
from mxnet import gluon
import gluonnlp as nlp
import gluonnlp.data.batchify as btf
from datasets.msvd import MSVD
from datasets.msrvtt import MSRVTT


class TrainValDataTransform:
    """Transform the machine translation dataset.

    Clip source and the target sentences to the maximum length. For the source sentence, append the
    EOS. For the target sentence, append BOS and EOS.

    Parameters
    ----------
    src_vocab : Vocab
    tgt_vocab : Vocab
    src_max_len : int
    tgt_max_len : int
    """

    def __init__(self, vocab, tgt_max_len=-1):
        self._vocab = vocab
        self._tgt_max_len = tgt_max_len

    def __call__(self, tgt):
        # For tgt_max_len < 0, we do not clip the sequence
        if self._tgt_max_len >= 0:
            tgt_sentence = self._vocab[tgt.split()[:self._tgt_max_len]]
        else:
            tgt_sentence = self._vocab[tgt.split()]

        tgt_sentence.insert(0, self._vocab[self._vocab.bos_token])
        tgt_sentence.append(self._vocab[self._vocab.eos_token])
        tgt_npy = np.array(tgt_sentence, dtype=np.int32)
        return tgt_npy


def load_translation_data(cfg):
    """Load translation dataset"""

    if cfg.DATA.DATASET == 'MSVD':
        cfg.DATA.SPLIT = 'train'
        data_train = MSVD(cfg)
        cfg.DATA.SPLIT = 'val'
        data_val = MSVD(cfg)
        cfg.DATA.SPLIT = 'test'
        data_test = MSVD(cfg)
        cfg.DATA.SPLIT = 'train'
    elif cfg.DATA.DATASET == 'MSRVTT':
        cfg.DATA.SPLIT = 'train'
        data_train = MSRVTT(cfg)
        cfg.DATA.SPLIT = 'val'
        data_val = MSRVTT(cfg)
        cfg.DATA.SPLIT = 'test'
        data_test = MSRVTT(cfg)
        cfg.DATA.SPLIT = 'train'
    else:
        raise NotImplementedError

    counter = nlp.data.count_tokens(data_train.get_word_list())  # text_data is a list of word occurences, allowing repetition
    vocab = nlp.Vocab(counter)
    data_train.vocab = vocab

    # use standard embeddings... we will learn our own (see https://gluon-nlp.mxnet.io/api/notes/vocab_emb.html for more details)
    # fasttext = nlp.embedding.create('fasttext', source='wiki.simple')
    # vocab.set_embedding(fasttext)

    data_train.transform = TrainValDataTransform(vocab, cfg.DATA.TGT_MAX_LEN)
    data_val.transform = TrainValDataTransform(vocab)
    data_test.transform = TrainValDataTransform(vocab)

    return data_train, data_val, data_test, data_val.get_captions(), data_test.get_captions(), vocab


def get_dataloader(data_set, cfg, dataset_type, use_average_length=False, num_shards=0, num_workers=8):
    """Create data loaders for training/validation/test."""
    assert dataset_type in ['train', 'val', 'test']

    if cfg.DATA.BUCKET_SCHEME == 'constant':
        bucket_scheme = nlp.data.ConstWidthBucket()
    elif cfg.DATA.BUCKET_SCHEME == 'linear':
        bucket_scheme = nlp.data.LinearWidthBucket()
    elif cfg.DATA.BUCKET_SCHEME == 'exp':
        bucket_scheme = nlp.data.ExpWidthBucket(bucket_len_step=1.2)
    else:
        raise NotImplementedError

    data_lengths = data_set.get_seq_lengths()
    if dataset_type != 'train':
        data_lengths = list(map(lambda x: x[-1], data_lengths))

    batchify_fn = btf.Tuple(btf.Pad(pad_val=0), btf.Pad(pad_val=0), btf.Stack(dtype='float32'), btf.Stack(dtype='float32'), btf.Stack())

    batch_sampler = nlp.data.FixedBucketSampler(lengths=data_lengths,
                                                batch_size=(cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE),
                                                num_buckets=cfg.DATA.NUM_BUCKETS,
                                                ratio=cfg.DATA.BUCKET_RATIO,
                                                shuffle=(dataset_type == 'train'),
                                                use_average_length=use_average_length,
                                                num_shards=num_shards,
                                                bucket_scheme=bucket_scheme)

    if dataset_type == 'train':
        logging.info('Train Batch Sampler:\n%s', batch_sampler.stats())
    elif dataset_type == 'val':
        logging.info('Valid Batch Sampler:\n%s', batch_sampler.stats())
    else:
        logging.info('Test Batch Sampler:\n%s', batch_sampler.stats())

    data_loader = gluon.data.DataLoader(data_set,
                                        batch_sampler=batch_sampler,
                                        batchify_fn=batchify_fn,
                                        num_workers=num_workers)

    return data_loader


def make_dataloader(data_train, data_val, data_test, cfg, use_average_length=False, num_shards=0, num_workers=8):

    """Create data loaders for training/validation/test."""
    train_data_loader = get_dataloader(data_train, cfg, dataset_type='train',
                                       use_average_length=use_average_length,
                                       num_shards=num_shards,
                                       num_workers=num_workers)

    val_data_loader = get_dataloader(data_val, cfg, dataset_type='val',
                                     use_average_length=use_average_length,
                                     num_workers=num_workers)

    test_data_loader = get_dataloader(data_test, cfg, dataset_type='test',
                                      use_average_length=use_average_length,
                                      num_workers=num_workers)

    return train_data_loader, val_data_loader, test_data_loader


def write_sentences(sentences, file_path):
    with io.open(file_path, 'w', encoding='utf-8') as of:
        for sent in sentences:
            if isinstance(sent, (list, tuple)):
                of.write(' '.join(sent) + '\n')
            else:
                of.write(sent + '\n')