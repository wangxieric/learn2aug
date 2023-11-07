from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import torch.nn as nn
import numpy as np
import os
import random
import time
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
import argparse
from math import ceil
from utils import write_log


'''
    Parameters
'''

parser = argparse.ArgumentParser()

# model information
parser.add_argument('--model_name', type=str, default='simpletod', help='model name')
parser.add_argument('--model_save_path', type=str, default='output', help='path to save model')
parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')

# training parameters
parser.add_argument('--eos_token_id', type=int, default=None, help='eos token id')
parser.add_argument('--max_len', type=int, default=512, help='max length of input sequence')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--learning rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--neg_sample', type=bool, default=True, help='whether to use negative sampling')
parser.add_argument('--neg_sample_rate', type=int, default=3, help='rate of nochitchat to chitchat')
parser.add_argument('--no_cuda', action='store_true', help='not use cuda')

# data path
parser.add_argument('--train_input', type=str, help='train input text file, instance separated by newline')
parser.add_argument('--dev_input', type=str, help='dev input text file, instance separated by newline')
parser.add_argument('--output', type=str, help='path to save model')

# analysis
parser.add_argument('--report_loss', type=int, default=100, help='report loss every n steps')
parser.add_argument('--save_model', type=int, default=1000, help='save model every n steps')

args = parser.parse_args()

args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
args. n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

model_dir_name = args.model_name + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_save_path = os.path.join(args.model_save_path, model_dir_name)
results_save_path = os.path.join(args.model_save_path, 'results')
saved_model_path = os.path.join(args.model_save_path, 'saved_model')

os.makedirs(results_save_path, exist_ok=False)
os.makedirs(saved_model_path, exist_ok=False)
log_file_dir = os.path.join(results_save_path, 'log_%s.txt' % model_dir_name)

# set seeds
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# set seed
set_seed(args)

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', pad_token='<PAD>')

tokenizer.add_special_tokens({'additional_special_tokens': ['<|belief|>', '<|endofbelief|>', '<|action|>', '<|endofaction|>', \
                                                            '<|response|>', '<|endofresponse|>', '<|context|>', '<|endofcontext|>', '<|user|>', '<|system|>', \
                                                            '<|task|>', '<|endoftask|>', '<|chitchat|>', '<|nochitchat|>', '<|endofdecision|>', '<|knowledge|>', \
                                                            '<|endofknowledge|>', '<|dbresults|>', '<|endofdbresults|>']})

model.resize_token_embeddings(len(tokenizer))
model = nn.DataParallel(model)
model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
model.train()

def language_modelling(model, conv_text, dev=True):
    encoding_dict = tokenizer.batch_encode_plus(conv_text, padding=True)
    input_ids = torch.tensor(encoding_dict['input_ids'])
    attention_mask = torch.tensor(encoding_dict['attention_mask'])

    seq_len = len(input_ids[0])

    if seq_len > args.max_len:
        input_ids =  torch.split(input_ids, args.max_len, dim=1)[0]
        attention_mask = torch.split(attention_mask, args.max_len, dim=1)[0]
        seq_len = len(input_ids[0])

    last_non_masked_idx =  torch.sum(attention_mask, dim=1) - 1

    position_ids = torch.tensor([list(range(seq_len)) for _ in range(len(input_ids))])
    for i, position_ids_slice in enumerate(position_ids):
        position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]
    
    input_ids = input_ids.to(args.device)
    attention_mask = attention_mask.to(args.device)
    position_ids = position_ids.to(args.device)

    # one-position shift happen automatically for GPT-2 while computing loss
    # so a shift is not needed here
    labels = input_ids.to(args.device)

    outputs = model(input_ids=input_ids, position_ids=position_ids, labels=labels, return_dict=True)
    loss = outputs.loss.sum()

    return loss

with open(args.train_input, 'r') as f:
    train_data = f.read().strip().split('\n')
    print('Number of training instances: %d' % len(train_data))

with open(args.dev_input, 'r') as f:
    dev_data = f.read().strip().split('\n')
    print('Number of dev instances: %d' % len(dev_data))

conv_chitchat = []
conv_no_chitchat = []
for conv in train_data:
    if '<|chitchat|>' in conv:
        conv_chitchat.append(conv)
    else:
        conv_no_chitchat.append(conv)

print('Number of chitchat instances: %d' % len(conv_chitchat))
print('Number of nochitchat instances: %d' % len(conv_no_chitchat))
num_train = len(conv_chitchat) * (args.neg_sample_rate + 1)

batch_size = args.batch_size
num_train_batch = ceil(num_train / batch_size)

num_dev_batch = ceil(len(dev_data) / batch_size)

start_time = time.time()
k = 0
record_k = 0
record_loss = 0.0

#  track input parameters
write_log(log_file=log_file_dir, s='#################### Input Parameters ####################')
for arg in vars(args):
    write_log(log_file=log_file_dir, s='{}: {}'.format(arg, getattr(args, arg)))
write_log(log_file=log_file_dir, s='##########################################################')

# train model
for _ in range(args.num_epochs):
    random.shuffle(conv_no_chitchat)
    if args.neg_sample:
        convs = conv_chitchat + conv_no_chitchat[:len(conv_chitchat) * args.neg_sample_rate]
    else:
        convs = conv_chitchat + conv_no_chitchat
    random.shuffle(convs)

    for batch in tqdm(range(num_train_batch)):
        conv_text = convs[batch * batch_size: (batch + 1) * batch_size]

        model.zero_grad()
        optimizer.zero_grad()

        loss = language_modelling(model, conv_text, dev=False)
        loss.backward()
        optimizer.step()

        record_loss += loss.item()
        k += 1
        record_k += 1

        if k > 1 and k % args.report_loss == 0:
            write_log(log_file_dir, "%d: loss = %.3f" % (k, record_loss / record_k))
            record_loss = 0.0
            record_k = 0
        
        if k > 1 and k % args.save_model == 0:
            print("Round: %d, saving model..." % k / args.save_model)
            model.eval()

            cost_time = time.time() - start_time
            write_log(log_file_dir, "Round: %d, cost time: %.3f" % 
                      (k // args.save_model, cost_time))
            start_time = time.time()

            if k // args.save_model >= 1:
                print("validating...")

                saved_model_path_cnt = os.path.join(saved_model_path, 'loads_%d' % (k // args.save_model))
                os.makedirs(saved_model_path_cnt, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(saved_model_path_cnt, '/model.pt'))

                eval_loss = 0.0
                with torch.no_grad():
                    for dev_batch in tqdm(range(num_dev_batch)):
                        dev_conv_text = dev_data[dev_batch * batch_size: (dev_batch + 1) * batch_size]
                        eval_loss += language_modelling(model, dev_conv_text).item()

                write_log(log_file_dir, "Round: %d, eval loss: %.3f" % eval_loss)

            model.train() 