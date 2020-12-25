import os
import numpy as np
from tqdm import tqdm
import logging
import random

import torch
import torch.distributed as dist
from transformers import BartTokenizer, BartConfig, GPT2TokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup

from data import QAData
from unified_data import UnifiedQAData
from bart import MyBart
from gpt2 import MyGPT2

def run(gpu, args):
    rank = gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_filename = "{}log-gpu{}.txt".format("" if args.do_train else "eval_", gpu)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    if args.lm_format:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    else:
        tokenizer = BartTokenizer.from_pretrained("bart-large")

    if args.is_unifiedqa:
        dev_data = UnifiedQAData(logger, args, args.predict_file, is_training=False, lm_format=args.lm_format)
    else:
        dev_data = QAData(logger, args, args.predict_file, False)

    if not args.skip_inference:
        dev_data.load_dataset(tokenizer)
        dev_data.load_dataloader()

    if args.do_train:
        if args.is_unifiedqa:
            train_data = UnifiedQAData(logger, args, args.train_file, is_training=True, lm_format=args.lm_format)
        else:
            train_data = QAData(logger, args, args.train_file, True)
        train_data.load_dataset(tokenizer)
        sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.world_size, rank=rank)
        train_data.load_dataloader(sampler)

        if args.checkpoint is not None:
            if args.lm_format:
                model = MyGPT2.from_pretrained("gpt2-large", state_dict=torch.load(args.checkpoint))
                #model.parallelize()
            else:
                model = MyBart.from_pretrained("bart-large", state_dict=torch.load(args.checkpoint))
        else:
            if args.lm_format:
                model = MyGPT2.from_pretrained("gpt2-large")
                #model.parallelize()
            else:
                model = MyBart.from_pretrained("bart-large")
        '''
        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)
        if args.n_gpu>0:
            model.to(torch.device("cuda"))
        '''
        torch.cuda.set_device(gpu)
        model.cuda(gpu)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.steps)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        train(args, logger, model, train_data, dev_data, optimizer, scheduler, rank)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, 'best-model.pt') if args.checkpoint is None else args.checkpoint
        model = MyBart.from_pretrained("bart-large",
                                       state_dict=torch.load(checkpoint))
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if args.n_gpu>0:
            model.to(torch.device("cuda"))
        model.eval()
        ems = inference(model, dev_data, save_predictions=True)
        logger.info("%s on %s data: %.2f" % (dev_data.metric, dev_data.data_type, np.mean(ems)*100))

def train(args, logger, model, train_data, dev_data, optimizer, scheduler, rank):
    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False

    if args.checkpoint_step > 0:
        for _ in range(args.checkpoint_step):
            global_step += 1
            scheduler.step()

    def convert_to_single_gpu(state_dict):
        def _convert(key):
            if key.startswith('module.'):
                return key[7:]
            return key
        return {_convert(key):value for key, value in state_dict.items()}

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_data.dataloader, disable=rank != 0):
            global_step += 1
            batch = [b.to(torch.device("cuda")) for b in batch]
            if args.lm_format:
                loss = model(input_ids=batch[0], attention_mask=batch[1], loss_mask=batch[2], labels=batch[0])
            else:
                loss = model(input_ids=batch[0], attention_mask=batch[1],
                             decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                             is_training=True)
            '''
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            '''
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            if global_step >= args.steps:
                logger.info("Stop training at {}" % global_step)
                stop_training = True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()

            if global_step % args.eval_period == 0 and rank == 0:
                if args.skip_inference:
                    logger.info("Step %d (epoch %d) Train loss %.2f" % (global_step, epoch, np.mean(train_losses)))
                    train_losses = []
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    if args.n_gpu > 1:
                        model_state_dict = convert_to_single_gpu(model_state_dict)
                    torch.save(model_state_dict, os.path.join(args.output_dir, "best-model-{}.pt".format(str(global_step).zfill(6))))
                else:
                    model.eval()
                    curr_em = inference(model if args.n_gpu==1 else model.module, dev_data)
                    logger.info("Step %d Train loss %.2f %s %.2f%% on epoch=%d" % (
                            global_step,
                            np.mean(train_losses),
                            dev_data.metric,
                            curr_em*100,
                            epoch))
                    train_losses = []
                    if best_accuracy < curr_em:
                        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                        if args.n_gpu > 1:
                            model_state_dict = convert_to_single_gpu(model_state_dict)
                        torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                        logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
                                (dev_data.metric, best_accuracy*100.0, curr_em*100.0, epoch, global_step))
                        best_accuracy = curr_em
                        wait_step = 0
                        stop_training = False
                    else:
                        wait_step += 1
                        if wait_step >= args.wait_step:
                            stop_training = True
                            break
                model.train()
        if stop_training:
            break

def inference(model, dev_data, save_predictions=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 min_lnegth=1,
                                 max_length=dev_data.args.max_output_length,
                                 early_stopping=True,)
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
    if save_predictions:
        dev_data.save_predictions(predictions)
    return np.mean(dev_data.evaluate(predictions))







