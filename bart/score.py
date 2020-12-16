from typing import List, Tuple, Dict, Any
import argparse
from collections import defaultdict
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizer, AdamW, get_linear_schedule_with_warmup
from bart import MyBart
CLEAN_TEST_DOMAINS = [('arc_hard', ('train', 'dev', 'test')),
                      ('ai2_science_middle', ('train', 'dev', 'test')),
                      ('mctest_corrected_the_separator', ('train', 'dev')),
                      ('social_iqa', ('train', 'dev')),
                      ('race_string', ('train', 'dev', 'test'))]
CLEAN_TRAIN_DOMAINS = [('arc_easy', ('train', 'dev', 'test')),
                       ('ai2_science_elementary', ('train', 'dev', 'test')),
                       ('openbookqa', ('train', 'dev', 'test')),
                       ('qasc', ('train', 'dev', 'test')),
                       ('winogrande_l', ('train', 'dev')),
                       ('commonsenseqa', ('train', 'dev', 'test')),
                       ('physical_iqa', ('train', 'dev', 'test'))]
SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def read_qa_data(data_dir: str, domains: List, split: str, format: str='tsv', has_ret: bool=False, use_inp: bool=False, load: bool=False):
  def agg(arr, item, prepend: bool):
    if prepend:
      arr.insert(0, item)
    else:
      arr.append(item)
  datas: Dict[Any, List] = defaultdict(list)
  for domain, splits in domains:
    file = os.path.join(data_dir, domain, split + '.' + format)
    with open(file, 'r') as fin:
      for l in fin:
        if has_ret:
          ls = l.rstrip('\n').split('\t')
          lid, question, answer, correct = ls[:4]
          lid = domain + '-' + lid
          question = question.replace('\\n', '\n')
          rets = ls[4:]
          qrets = rets[:len(rets) // 2]
          question = question + ' \n ' + qrets[0]
          if use_inp:
            sample = (lid, '', question)
            if load:
              agg(datas[lid], sample, correct == 'True')
            else:
              yield sample
          else:
            sample = (lid, question, answer)
            if load:
              agg(datas[lid], sample, correct == 'True')
            else:
              yield sample
        else:
          ls = l.rstrip('\n').split('\t')
          if len(ls) == 4:
            lid, question, answer, correct = ls
            lid = domain + '-' + lid
          elif len(ls) == 2:
            question, answer = ls
            lid = 0
          else:
            raise NotImplementedError
          question = question.replace('\\n', '\n')
          if use_inp:
            sample = (lid, '', question)
            if load:
              agg(datas[lid], sample, correct == 'True')
            else:
              yield sample
          else:
            sample = (lid, question, answer)
            if load:
              agg(datas[lid], sample, correct == 'True')
            else:
              yield sample
  if load:
    keys = sorted(datas.keys())
    print('total number of questions {}'.format(len(datas)))
    perm = np.random.permutation(len(datas))
    for p in perm:
      yield from datas[keys[p]]


def string_to_tensor(tokenizer,
                     data,
                     max_input_len: int,
                     max_target_len: int,
                     max_token_per_batch: int=None,
                     append_bos: bool=False,
                     num_options: int=0,
                     device: str='cuda') -> Tuple[Dict[str, torch.Tensor], List[int]]:
  combine = {'input': [], 'input_mask': [], 'target': [], 'target_mask': [], 'sample_ind': []}
  max_inp_len = max_tar_len = 0
  n_question = -1
  n_option = 0
  prev_id = None
  for id, input, target in data:
    if append_bos:
      input, target = '<s> ' + input, '<s> ' + target
    # tokenize
    t_input = tokenizer.batch_encode_plus([input], max_length=max_input_len)  # <s> </s>
    t_target = tokenizer.batch_encode_plus([target], max_length=max_target_len)  # <s> </s>
    # truncate
    inp = t_input['input_ids'][0]
    tar = t_target['input_ids'][0]
    inp_att = t_input['attention_mask'][0]
    tar_att = t_target['attention_mask'][0]
    # yield or not
    if max(len(inp), max_inp_len) * (len(combine['input']) + 1) + \
      max(len(tar), max_tar_len) * (len(combine['target']) + 1) > max_token_per_batch and (not num_options or id != prev_id):
      combine['input'] = pad_sequence([torch.LongTensor(i) for i in combine['input']], batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
      combine['input_mask'] = pad_sequence([torch.LongTensor(i) for i in combine['input_mask']], batch_first=True, padding_value=0).to(device)
      combine['target'] = pad_sequence([torch.LongTensor(i) for i in combine['target']], batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
      combine['target_mask'] = pad_sequence([torch.LongTensor(i) for i in combine['target_mask']], batch_first=True, padding_value=0).to(device)
      combine['sample_ind'] = torch.LongTensor(combine['sample_ind']).to(device)
      yield n_question + 1, combine
      combine = {'input': [], 'input_mask': [], 'target': [], 'target_mask': [], 'sample_ind': []}
      max_inp_len = max_tar_len = 0
      n_question = -1
      n_option = 0
    # save
    combine['input'].append(inp)
    combine['input_mask'].append(inp_att)
    combine['target'].append(tar)
    combine['target_mask'].append(tar_att)
    if prev_id == id:
      n_option += 1
    else:
      n_question += 1
      n_option = 0
    prev_id = id
    combine['sample_ind'].append(n_question * num_options + n_option)
    max_inp_len = max(max_inp_len, len(inp))
    max_tar_len = max(max_tar_len, len(tar))
  if len(combine['input']) > 0:
    combine['input'] = pad_sequence([torch.LongTensor(i) for i in combine['input']], batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    combine['input_mask'] = pad_sequence([torch.LongTensor(i) for i in combine['input_mask']], batch_first=True, padding_value=0).to(device)
    combine['target'] = pad_sequence([torch.LongTensor(i) for i in combine['target']], batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    combine['target_mask'] = pad_sequence([torch.LongTensor(i) for i in combine['target_mask']], batch_first=True, padding_value=0).to(device)
    combine['sample_ind'] = torch.LongTensor(combine['sample_ind']).to(device)
    yield n_question + 1, combine


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='score targets using models from transformers')
  parser.add_argument('--task', type=str, choices=['score', 'softmax', 'margin'], default='score')
  parser.add_argument('--model', type=str, default='unifiedQA-uncased/best-model.pt')
  parser.add_argument('--data', type=str, help='path to the data')
  parser.add_argument('--num_options', type=int, help='max number of candidate answers', default=0)
  parser.add_argument('--domains', type=str, default='clean_test_domains')
  parser.add_argument('--split', type=str, help='split', default='dev')
  parser.add_argument('--output', type=str, help='output file')
  parser.add_argument('--max_token_per_batch', type=int, default=5000)
  parser.add_argument('--lr', type=float, default=1e-5)
  parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='epsilon for Adam optimizer')
  parser.add_argument('--warmup_steps', default=0, type=int, help='linear warmup over warmup_steps')
  parser.add_argument('--weight_decay', default=0.0, type=float, help='weight deay if we apply some')
  parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max gradient norm')
  parser.add_argument('--steps', type=int, default=15000)
  parser.add_argument('--has_ret', action='store_true')
  parser.add_argument('--use_inp', action='store_true')
  args = parser.parse_args()

  device = 'cuda'
  max_input_len = 512
  max_target_len = 128
  base_model = 'facebook/bart-large'
  append_bos = True
  if args.task == 'score':
    args.max_token_per_batch = 5000
  else:
    args.max_token_per_batch = 5000

  print('init data')
  tokenizer = BartTokenizer.from_pretrained(base_model)
  def get_iter():
    data = read_qa_data(args.data, eval(args.domains.upper()), split=args.split, has_ret=args.has_ret, use_inp=args.use_inp, load=args.num_options > 0)
    iter = string_to_tensor(tokenizer, data, max_input_len=max_input_len, max_target_len=max_target_len, max_token_per_batch=args.max_token_per_batch, append_bos=append_bos, num_options=args.num_options, device=device)
    return iter

  print('loading models ...')
  if args.model == 'facebook/bart-large':
    model = MyBart.from_pretrained(base_model).to(device)
  else:
    model = MyBart.from_pretrained(base_model, state_dict=torch.load(args.model)).to(device)

  os.makedirs(os.path.dirname(args.output), exist_ok=True)
  if args.task == 'score':
    model.eval()
    iter = get_iter()
    pbar = tqdm()
    with open(args.output, 'w') as fout:
      for _, input_dict in iter:
        logprobs = model(input_ids=input_dict['input'], attention_mask=input_dict['input_mask'],
                         decoder_input_ids=input_dict['target'], decoder_attention_mask=input_dict['target_mask'],
                         is_training=True, return_logprob=True)
        for inp, tar, tar_mask, lp in zip(input_dict['input'].cpu().numpy(), input_dict['target'].cpu().numpy(), input_dict['target_mask'].cpu().numpy(), logprobs.cpu().numpy()):
          score = np.sum(lp * tar_mask)
          fout.write('{}\t{}\t{}\t{}\n'.format(score, ','.join(map(str, inp)), ','.join(map(str, tar)), ','.join(map(str, lp))))
          pbar.update(1)
    pbar.close()
  else:
    model.train()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
      {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.steps)
    global_step = 0
    #pbar = tqdm()
    losses = []
    while True:
      iter = get_iter()
      for num_question, input_dict in iter:
        loss = model(input_ids=input_dict['input'], attention_mask=input_dict['input_mask'],
                     decoder_input_ids=input_dict['target'], decoder_attention_mask=input_dict['target_mask'],
                     is_training=True, sample_ind=input_dict['sample_ind'], objective=args.task,
                     num_question=num_question, num_options=args.num_options)
        loss.backward()
        losses.append(loss.detach().cpu().numpy())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        global_step += 1
        #pbar.update(1)
        if global_step % 100 == 0:
          print('step {}, loss {}'.format(global_step, np.mean(losses[-100:])), flush=True)
        if global_step % 2000 == 0:
          model_state_dict = {k: v.cpu() for (k, v) in model.state_dict().items()}
          torch.save(model_state_dict, '{}.{}'.format(args.output, global_step))
          print('save at {}'.format(global_step), flush=True)
        if global_step >= args.steps:
          break
      if global_step >= args.steps:
        break
    model_state_dict = {k: v.cpu() for (k, v) in model.state_dict().items()}
    torch.save(model_state_dict, '{}.{}'.format(args.output, global_step))
    #pbar.close()
