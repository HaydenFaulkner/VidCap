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
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import logging
import math
import os
import pprint
import random
import time

import numpy as np
import mxnet as mx
from mxnet import gluon

import gluonnlp as nlp
from gluonnlp.loss import LabelSmoothing, MaskedSoftmaxCELoss
from gluonnlp.model.transformer import ParallelTransformer, get_transformer_encoder_decoder
from gluonnlp.model.transformer import NMTModel
from gluonnlp.utils.parallel import Parallel
from mxnet.gluon import nn
import utils.dataprocessor as dataprocessor
from metrics.bleu import compute_bleu
from utils.beam_search import BeamSearchTranslator
from config import config, update_config
# from models.transformer import NMTModel

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

nlp.utils.check_version('0.9.0')
from mxnet.gluon import HybridBlock


class TimeDistributed(HybridBlock):
    def __init__(self, model, style='reshape', **kwargs):
        """
        A time distributed layer like that seen in Keras
        Args:
            model: the backbone model that will be repeated over time
            style (str): either 'reshape' or 'for' for the implementation to use (default is reshape1)
        """
        super(TimeDistributed, self).__init__(**kwargs)
        assert style in ['reshape', 'for']

        self._style = style
        with self.name_scope():
            self.model = model

    def apply_model(self, x, _):
        return self.model(x), []

    def hybrid_forward(self, F, x):
        if self._style == 'for':
            # For loop style
            x = F.swapaxes(x, 0, 1)  # swap batch and seqlen channels
            x, _ = F.contrib.foreach(self.apply_model, x, [])  # runs on first channel, which is now seqlen
            if isinstance(x, tuple):  # for handling multiple outputs
                x = (F.swapaxes(xi, 0, 1) for xi in x)
            elif isinstance(x, list):
                x = [F.swapaxes(xi, 0, 1) for xi in x]
            else:
                x = F.swapaxes(x, 0, 1)  # swap seqlen and batch channels
        else:
            shp = x  # can use this to keep shapes for reshape back to (batch, timesteps, ...)
            x = F.reshape(x, (-3, -2))  # combines batch and timesteps dims
            x = self.model(x)
            if isinstance(x, tuple):  # for handling multiple outputs
                x = (F.reshape_like(xi, shp, lhs_end=1, rhs_end=2) for xi in x)
            elif isinstance(x, list):
                x = [F.reshape_like(xi, shp, lhs_end=1, rhs_end=2) for xi in x]
            else:
                x = F.reshape_like(x, shp, lhs_end=1, rhs_end=2)  # (num_samples, timesteps, ...)

        return x

def parse_args():

    parser = argparse.ArgumentParser(description='Train Transformer Video Captioner')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()
    head = '%(asctime)-15s %(message)s'
    config.SAVE_DIR = os.path.join('runs', os.path.basename(args.cfg[:-5]))
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    logging.basicConfig(filename=os.path.join(config.SAVE_DIR, 'training_log.txt'), format=head)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    train(config, logger)


def evaluate(ctx, data_loader, model, test_loss_function, translator, vocab):
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    translation_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0
    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) in enumerate(data_loader):
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)
        # Calculating Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = test_loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)
        # Translate
        samples, _, sample_valid_length = translator.translate(src_seq=src_seq, src_valid_length=src_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()
        for i in range(max_score_sample.shape[0]):
            translation_out.append([vocab.idx_to_token[ele] for ele in max_score_sample[i][1:(sample_valid_length[i] - 1)]])

    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]

    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = sentence

    return avg_loss, real_translation_out


def train(cfg, logger):
    # load datasets
    data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, vocab = dataprocessor.load_translation_data(cfg)

    dataprocessor.write_sentences(val_tgt_sentences, os.path.join(cfg.SAVE_DIR, 'val_gt.txt'))
    dataprocessor.write_sentences(test_tgt_sentences, os.path.join(cfg.SAVE_DIR, 'test_gt.txt'))

    # data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
    # data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
    #                                      for i, ele in enumerate(data_val)])
    # data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
    #                                       for i, ele in enumerate(data_test)])

    data_train_l, data_val_l, data_test_l = [d.get_seq_lengths() for d in [data_train, data_val, data_test]]

    if cfg.DATA.SRC_MAX_LEN <= 0 or cfg.DATA.TGT_MAX_LEN <= 0:
        max_len = np.max([np.max(data_train_l, axis=0), np.max(data_val_l, axis=0), np.max(data_test_l, axis=0)], axis=0)

    if cfg.DATA.SRC_MAX_LEN > 0:
        src_max_len = cfg.DATA.SRC_MAX_LEN
    else:
        src_max_len = max_len[0]

    if cfg.DATA.TGT_MAX_LEN > 0:
        tgt_max_len = cfg.DATA.TGT_MAX_LEN
    else:
        tgt_max_len = max_len[1]

    # get contexts GPU or CPU
    ctx = [mx.cpu()] if len(cfg.GPU) < 1 else [mx.gpu(int(x)) for x in cfg.GPU]
    num_ctxs = len(ctx)

    # get the model, do the encoder/decoder
    encoder, decoder, one_step_ahead_decoder = get_transformer_encoder_decoder(
        units=cfg.MODEL.EMB_SIZE, hidden_size=cfg.MODEL.HIDDEN_SIZE, dropout=cfg.MODEL.DROPOUT,
        num_layers=cfg.MODEL.NUM_LAYERS, num_heads=cfg.MODEL.NUM_HEADS, max_src_length=max(src_max_len, 500),
        max_tgt_length=max(tgt_max_len, 500), scaled=cfg.MODEL.SCALED_ATT)

    src_embed = nn.HybridSequential(prefix='src_embed_')
    with src_embed.name_scope():
        src_embed.add(TimeDistributed(nn.Dense(units=cfg.MODEL.EMB_SIZE)))
        src_embed.add(nn.Dropout(rate=0.0))

    model = NMTModel(src_vocab=None, tgt_vocab=vocab, encoder=encoder, decoder=decoder,
                     one_step_ahead_decoder=one_step_ahead_decoder, embed_size=cfg.MODEL.EMB_SIZE,
                     tie_weights=False, embed_initializer=None, prefix='transformer_', src_embed=src_embed)

    model.initialize(init=mx.init.Xavier(magnitude=cfg.TRAIN.MAGNITUDE), ctx=ctx)
    static_alloc = True
    model.hybridize(static_alloc=static_alloc)
    logger.info(model)

    # make the translator
    translator = BeamSearchTranslator(model=model, beam_size=cfg.TRAIN.BEAM_SIZE,
                                      scorer=nlp.model.BeamSearchScorer(alpha=cfg.TRAIN.LP_ALPHA,
                                                                        K=cfg.TRAIN.LP_K),
                                      max_length=200)

    logger.info('Use beam_size={}, alpha={}, K={}'.format(cfg.TRAIN.BEAM_SIZE, cfg.TRAIN.LP_ALPHA, cfg.TRAIN.LP_K))

    # init the losses and label smoothing
    label_smoothing = LabelSmoothing(epsilon=cfg.TRAIN.EPSILON, units=len(vocab))
    label_smoothing.hybridize(static_alloc=static_alloc)

    loss_function = MaskedSoftmaxCELoss(sparse_label=False)
    loss_function.hybridize(static_alloc=static_alloc)

    test_loss_function = MaskedSoftmaxCELoss()
    test_loss_function.hybridize(static_alloc=static_alloc)

    rescale_loss = 100.
    parallel_model = ParallelTransformer(model, label_smoothing, loss_function, rescale_loss)

    trainer = gluon.Trainer(model.collect_params(), cfg.TRAIN.OPTIMIZER,
                            {'learning_rate': cfg.TRAIN.LR, 'beta2': 0.98, 'epsilon': 1e-9})

    train_data_loader, val_data_loader, test_data_loader \
        = dataprocessor.make_dataloader(data_train, data_val, data_test, cfg, use_average_length=True, num_shards=len(ctx))

    assert len(ctx) < 2, "cant handle more than 1 gpu as it breaks the batching ops"

    bpe = True
    split_compound_word = bpe
    tokenized = True

    # training loop
    best_valid_bleu = 0.0
    step_num = 0
    warmup_steps = cfg.TRAIN.WARMUP_STEPS
    grad_interval = cfg.TRAIN.NUM_ACCUMULATED
    model.collect_params().setattr('grad_req', 'add')
    average_start = (len(train_data_loader) // grad_interval) * (cfg.TRAIN.EPOCHS - cfg.TRAIN.AVG_START)
    average_param_dict = None
    model.collect_params().zero_grad()
    parallel = Parallel(num_ctxs, parallel_model)
    for epoch_id in range(cfg.TRAIN.EPOCHS):
        log_avg_loss = 0
        log_wc = 0
        loss_denom = 0
        step_loss = 0
        log_start_time = time.time()
        for batch_id, seqs in enumerate(train_data_loader):
            if batch_id % grad_interval == 0:
                step_num += 1
                new_lr = cfg.TRAIN.LR / math.sqrt(cfg.MODEL.EMB_SIZE) * min(1. / math.sqrt(step_num), step_num * warmup_steps ** (-1.5))
                trainer.set_learning_rate(new_lr)

            seqs = [seqs[:4]]  # my addition assuming gpu is 1, removes sample id
            src_wc, tgt_wc, bs = np.sum([(shard[2].sum(), shard[3].sum(), shard[0].shape[0]) for shard in seqs], axis=0)
            seqs = [[seq.as_in_context(context) for seq in shard] for context, shard in zip(ctx, seqs)]
            Ls = []
            for seq in seqs:
                parallel.put((seq, cfg.TRAIN.BATCH_SIZE))
            Ls = [parallel.get() for _ in range(len(ctx))]

            src_wc = src_wc.asscalar()
            tgt_wc = tgt_wc.asscalar()
            loss_denom += tgt_wc - bs
            if batch_id % grad_interval == grad_interval - 1 or batch_id == len(train_data_loader) - 1:
                if average_param_dict is None:
                    average_param_dict = {k: v.data(ctx[0]).copy() for k, v in model.collect_params().items()}

                trainer.step(float(loss_denom) / cfg.TRAIN.BATCH_SIZE / rescale_loss)
                param_dict = model.collect_params()
                param_dict.zero_grad()

                if step_num > average_start:
                    alpha = 1. / max(1, step_num - average_start)
                    for name, average_param in average_param_dict.items():
                        average_param[:] += alpha * (param_dict[name].data(ctx[0]) - average_param)

            step_loss += sum([L.asscalar() for L in Ls])
            if batch_id % grad_interval == grad_interval - 1 or\
                    batch_id == len(train_data_loader) - 1:
                log_avg_loss += step_loss / loss_denom * cfg.TRAIN.BATCH_SIZE * rescale_loss
                loss_denom = 0
                step_loss = 0
            log_wc += src_wc + tgt_wc

            if (batch_id + 1) % (cfg.LOG_INTERVAL * grad_interval) == 0:
                wps = log_wc / (time.time() - log_start_time)
                logger.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, '
                            'throughput={:.2f}K wps, wc={:.2f}K'
                            .format(epoch_id, batch_id + 1, len(train_data_loader),
                                    log_avg_loss / cfg.LOG_INTERVAL,
                                    np.exp(log_avg_loss / cfg.LOG_INTERVAL),
                                    wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_wc = 0

        mx.nd.waitall()

        valid_loss, valid_translation_out = evaluate(ctx[0], val_data_loader, model, test_loss_function, translator, vocab)

        valid_bleu_score, _, _, _, _ = compute_bleu([[v.split() for v in val_tgt_sentences]], valid_translation_out,
                                                    tokenized=tokenized, tokenizer='tweaked',
                                                    split_compound_word=split_compound_word,
                                                    bpe=bpe)
        logger.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                    .format(epoch_id, valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
        dataprocessor.write_sentences(valid_translation_out,
                                      os.path.join(cfg.SAVE_DIR, 'epoch{:d}_valid_out.txt').format(epoch_id))

        if valid_bleu_score > best_valid_bleu:
            best_valid_bleu = valid_bleu_score
            save_path = os.path.join(cfg.SAVE_DIR, 'valid_best.params')
            logger.info('Save best parameters to {}'.format(save_path))
            model.save_parameters(save_path)
        save_path = os.path.join(cfg.SAVE_DIR, 'epoch{:d}.params'.format(epoch_id))
        model.save_parameters(save_path)

    save_path = os.path.join(cfg.SAVE_DIR, 'average.params')
    mx.nd.save(save_path, average_param_dict)

    if cfg.TRAIN.AVG_CHECKPOINT:
        for j in range(cfg.TRAIN.NUM_AVERAGES):
            params = mx.nd.load(os.path.join(cfg.SAVE_DIR, 'epoch{:d}.params'.format(cfg.TRAIN.EPOCHS - j - 1)))
            alpha = 1. / (j + 1)
            for k, v in model._collect_params_with_prefix().items():
                for c in ctx:
                    v.data(c)[:] += alpha * (params[k].as_in_context(c) - v.data(c))

        save_path = os.path.join(cfg.SAVE_DIR, 'average_checkpoint_{}.params'.format(cfg.TRAIN.NUM_AVERAGES))
        model.save_parameters(save_path)

    elif cfg.TRAIN.AVG_START > 0:
        for k, v in model.collect_params().items():
            v.set_data(average_param_dict[k])
        save_path = os.path.join(cfg.SAVE_DIR, 'average.params')
        model.save_parameters(save_path)
    else:
        model.load_parameters(os.path.join(cfg.SAVE_DIR, 'valid_best.params'), ctx)

    valid_loss, valid_translation_out = evaluate(ctx[0], val_data_loader, model, test_loss_function, translator, vocab)

    valid_bleu_score, _, _, _, _ = compute_bleu([[v.split() for v in val_tgt_sentences]], valid_translation_out,
                                                tokenized=tokenized, tokenizer='tweaked', bpe=bpe,
                                                split_compound_word=split_compound_word)
    logger.info('Best model valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                .format(valid_loss, np.exp(valid_loss), valid_bleu_score * 100))

    # run on test set
    test_loss, test_translation_out = evaluate(ctx[0], test_data_loader, model, test_loss_function, translator, vocab)

    test_bleu_score, _, _, _, _ = compute_bleu([[v.split() for v in test_tgt_sentences]], test_translation_out,
                                               tokenized=tokenized, tokenizer='tweaked', bpe=bpe,
                                               split_compound_word=split_compound_word)
    logger.info('Best model test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
                .format(test_loss, np.exp(test_loss), test_bleu_score * 100))
    dataprocessor.write_sentences(valid_translation_out,
                                  os.path.join(cfg.SAVE_DIR, 'best_valid_out.txt'))
    dataprocessor.write_sentences(test_translation_out,
                                  os.path.join(cfg.SAVE_DIR, 'best_test_out.txt'))


if __name__ == '__main__':
    main()
