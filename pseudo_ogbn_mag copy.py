import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from block_dataloader import generate_dataloader
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import deepspeed
import random
from graphsage_model import SAGE

from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, load_mag
from memory_usage import see_memory_usage
import tracemalloc
from cpu_mem_usage import get_memory
# from utils import draw_graph_global


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.gpu >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True


def ttt(tic, str1):
	toc = time.time()
	print(str1 + ' step Time(s): {:.4f}'.format(toc - tic))
	return toc


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1)==labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, device):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	model.eval()
	with torch.no_grad():
		pred = model.inference(g, nfeat, device, args)
	model.train()
	return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	print('batch_inputs device')
	print(batch_inputs.device)
	return batch_inputs, batch_labels


def load_blocks_subtensor(g, labels, blocks, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	# print('\nblocks[0].srcdata dgl.NID list')
	# print(blocks[0].srcdata[dgl.NID].tolist())
	batch_inputs = g.ndata['features'][blocks[0].srcdata[dgl.NID].tolist()].to(device)
	batch_labels = blocks[-1].dstdata['labels'].to(device)
	# print('\nblocks[-1].dstdata')
	# print(blocks[-1].dstdata[dgl.NID])
	# print()
	# print('batch_inputs device')
	# print(batch_inputs.device)
	return batch_inputs, batch_labels


#### Entry point
def run(args, device, data, tic):
	# Unpack data
	n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
	val_nfeat, val_labels, test_nfeat, test_labels = data

	in_feats = train_nfeat.shape[1]
	train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
	val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
	test_nid = torch.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
	dataloader_device = torch.device('cpu')

	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])

	# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
	full_batch_size = len(train_nid)
	full_batch_dataloader = dgl.dataloading.NodeDataLoader(
		train_g,
		train_nid,
		sampler,

		batch_size=full_batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)

	model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)
	model = model.to(device)
	loss_fcn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	
	epoch_train_CPU_time_list = []
	# see_memory_usage("-----------------------------------------before for epoch loop ")
	iter_tput = []
	avg_step_data_trans_time_list = []
	avg_step_GPU_train_time_list = []
	avg_step_time_list = []
	block_generate_time_list = []


	
	for epoch in range(args.num_epochs):
		print('Epoch ' + str(epoch))
		# train_start_tic = time.time()
		
		# data loader sampling fan-out neighbor each new epoch
		for full_batch_step, (input_nodes, output_seeds, full_batch_blocks) in enumerate(full_batch_dataloader):

			# Create DataLoader for constructing blocks
			print('----main run function: start generate block_dataloader from full batch train graph')
			ssss_time = time.time()
			block_dataloader, weights_list = generate_dataloader(train_g, full_batch_blocks,  args)
			block_generate_time = time.time() - ssss_time
			print('----main run function: block dataloader generation total spend   ' + str(block_generate_time))
			block_generate_time_list.append(block_generate_time)
			
			# Training loop
			step_time_list = []
			step_data_trans_time_list = []
			step_GPU_train_time_list = []

			# Loop over the dataloader to sample the computation dependency graph as a list of blocks.
			# torch.cuda.synchronize()
			start = torch.cuda.Event(enable_timing=True)
			end = torch.cuda.Event(enable_timing=True)
			
			print('length of block dataloader')
			print(len(block_dataloader))

			pseudo_mini_loss = torch.tensor([], dtype=torch.long)
			loss_sum = 0
			train_start_tic = time.time()
			tic_step = time.time()

			for step, (input_node, seeds, blocks) in enumerate(block_dataloader):
				# print("\n   ***************************     step   " + str(step) + " mini batch block  *************************************")
				
				torch.cuda.synchronize()
				start.record()
				
				# Load the input features as well as output labels
				batch_inputs, batch_labels = load_blocks_subtensor(train_g, train_labels, blocks, device)
				blocks = [block.int().to(device) for block in blocks]

				end.record()
				torch.cuda.synchronize()  # wait for move to complete
				step_data_trans_time_list.append(start.elapsed_time(end))
				#----------------------------------------------------------------------------------------
				start1 = torch.cuda.Event(enable_timing=True)
				end1 = torch.cuda.Event(enable_timing=True)
				start1.record()

				# Compute loss and prediction
				batch_pred = model(blocks, batch_inputs)
				pseudo_mini_loss = loss_fcn(batch_pred, batch_labels)
				pseudo_mini_loss = pseudo_mini_loss*weights_list[step]
				pseudo_mini_loss.backward()
				loss_sum += pseudo_mini_loss

				end1.record()
				torch.cuda.synchronize()  # wait for all training steps to complete
				step_GPU_train_time_list.append(start1.elapsed_time(end1))

				step_time = time.time() - tic_step
				step_time_list.append(step_time)
				torch.cuda.empty_cache()
				# print(step_time)

				iter_tput.append(len(seeds) / (time.time() - tic_step))

				tic_step = time.time()

			# see_memory_usage("-----------------------------------------before final loss ")

			optimizer.step()
			optimizer.zero_grad()

			train_end_toc = time.time()

			epoch_train_CPU_time_list.append((train_end_toc - train_start_tic ))
			print('current Epoch training on CPU without block data loder Time(s): {:.4f}'.format(train_end_toc - train_start_tic ))
			# see_memory_usage("-----------------------------------------after optimizer.step() ")

			# print('full batch loss ' + str(full_batch_loss.tolist()))
			
			print('pseudo_mini_loss  ' + str(pseudo_mini_loss.tolist()))
			print('pseudo_mini_loss sum ' + str(loss_sum.tolist()))
			if epoch % args.log_every==0:
				acc = compute_acc(batch_pred, batch_labels)
				gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024 /1024 if torch.cuda.is_available() else 0
				print(
					'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.4f} GB'.format(
						epoch, 0, pseudo_mini_loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

			#
			#
			indent = 0
			if len(step_data_trans_time_list[indent:]) > 0:
				avg_iteration_time = sum(step_data_trans_time_list[indent:]) / len(step_data_trans_time_list[indent:])
				print('\t\tavg iteration(step) data from cpu to GPU time:%.8f ms' % (avg_iteration_time))
				avg_step_data_trans_time_list.append(avg_iteration_time)
			
				avg_iteration_gpu_time = sum(step_GPU_train_time_list[indent:]) / len(step_GPU_train_time_list[indent:])
				print('\t\tavg iteration GPU training time:%.8f ms' % (avg_iteration_gpu_time))
				avg_step_GPU_train_time_list.append(avg_iteration_gpu_time)
			
				avg_step_time = sum(step_time_list[indent:]) / len(step_time_list[indent:])
				print('\t\tavg iteration (step) total cpu time:%.8f ms' % (avg_step_time * 1000))
				avg_step_time_list.append(avg_step_time)
			
			
			# if epoch % args.eval_every == 0 and epoch != 0:
			# 	eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device)
			# 	print('\t\tEval Acc {:.4f}'.format(eval_acc))

		# print('Avg cpu epoch time: {} ms'.format(avg*1000 / (epoch - 4)))
	out_indent = 2 # skip the first 2 epochs, initial epoch time is not stable.
	if len(avg_step_data_trans_time_list[out_indent:]) > 0:
		
		total_avg_iteration_time = sum(avg_step_data_trans_time_list[out_indent:]) / len(avg_step_data_trans_time_list[out_indent:])
		print('\ttotal avg iteration(step) data from cpu to GPU time:%.8f s' % (total_avg_iteration_time/1000))
		# print(len(avg_step_data_trans_time_list))
	
		total_avg_iteration_gpu_time = sum(avg_step_GPU_train_time_list[out_indent:]) / len(avg_step_GPU_train_time_list[out_indent:])
		print('\ttotal avg iteration GPU training time:%.8f s' % (total_avg_iteration_gpu_time/1000))
		# print(len(avg_step_GPU_train_time_list))
		# print(avg_step_GPU_train_time_list)
	
		total_avg_step_time = sum(avg_step_time_list[out_indent:]) / len(avg_step_time_list[out_indent:])
		print('\ttotal avg iteration (step) total cpu time:%.8f s' % (total_avg_step_time ))
		# print(len(avg_step_time_list))

	avg_epoch_total_cpu_train_time = sum(epoch_train_CPU_time_list[out_indent:]) / len(epoch_train_CPU_time_list[out_indent:])
	print('\ntotal avg epoch training  total cpu time:%.8f s' % (avg_epoch_total_cpu_train_time ))
	# print(len(epoch_train_CPU_time_list[out_indent:]))
	avg_block_generate_time = sum(block_generate_time_list[out_indent:]) / len(block_generate_time_list[out_indent:])
	print('\ntotal avg block generate cpu time:%.8f second' % (avg_block_generate_time ))
	

	# train_acc = evaluate(model, train_g, train_nfeat, train_labels, train_nid, device)
	# print('train Acc: {:.4f}'.format(train_acc))
	# test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device)
	# print('Test Acc: {:.4f}'.format(test_acc))


if __name__=='__main__':
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--gpu', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)

	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--selection-method', type=str, default='range')
	argparser.add_argument('--num-epochs', type=int, default=6)
	argparser.add_argument('--num-hidden', type=int, default=16)
	argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='20')
	argparser.add_argument('--fan-out', type=str, default='10')


	# argparser.add_argument('--batch-size', type=int, default=196571)
	# argparser.add_argument('--batch-size', type=int, default=98308)
	# argparser.add_argument('--batch-size', type=int, default=49154)
	# argparser.add_argument('--batch-size', type=int, default=24577)
	# argparser.add_argument('--batch-size', type=int, default=12289)
	# argparser.add_argument('--batch-size', type=int, default=6145)
	# argparser.add_argument('--batch-size', type=int, default=3000)
	# argparser.add_argument('--batch-size', type=int, default=1500)
	argparser.add_argument('--batch-size', type=int, default=8)


	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)

	argparser.add_argument('--lr', type=float, default=0.003)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	argparser.add_argument('--inductive', action='store_true',
		help="Inductive learning setting")
	argparser.add_argument('--data-cpu', action='store_true',
		help="By default the script puts all node features and labels "
		     "on GPU when using it to save time for data copy. This may "
		     "be undesired if they cannot fit in GPU memory at once. "
		     "This flag disables that.")
	args = argparser.parse_args()

	if args.gpu >= 0:
		device = torch.device('cuda:%d' % args.gpu)
	else:
		device = torch.device('cpu')
	set_seed(args)

	# get_memory("-----------------------------------------before load_ogb***************************")
	t2 = ttt(tt, "before load_data")
	if args.dataset=='karate':
		g, n_classes = load_karate()
	elif args.dataset=='cora':
		g, n_classes = load_cora()
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset=='ogbn-mag':
		g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = load_mag(args.dataset)
		
	else:
		raise Exception('unknown dataset')
	# see_memory_usage("-----------------------------------------after data to cpu------------------------")
	t3 = ttt(t2, "after load_dataset")
	if args.inductive:
		train_g, val_g, test_g = inductive_split(g)
		train_nfeat = train_g.ndata.pop('features')
		val_nfeat = val_g.ndata.pop('features')
		test_nfeat = test_g.ndata.pop('features')
		train_labels = train_g.ndata.pop('labels')
		val_labels = val_g.ndata.pop('labels')
		test_labels = test_g.ndata.pop('labels')
	else:
		train_g = val_g = test_g = g
		train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('feat')
		print('g.ndata')
		print(g.ndata)
		train_labels = val_labels = test_labels = g.ndata.pop('label')
		print('train_labels------------------------------------------------------------------------------')
		print(train_labels)
		print(len(train_labels))

	# get_memory("-----------------------------------------after inductive else***************************")
	t4 = ttt(t3, "after inductive else")
	print('args.data_cpu')
	print(args.data_cpu)

	if not args.data_cpu:
		train_nfeat = train_nfeat.to(device)
		train_labels = train_labels.to(device)
	# get_memory("-----------------------------------------after label***************************")
	t5 = ttt(t4, "after label")
	# Create csr/coo/csc formats before launching training processes with multi-gpu.
	# This avoids creating certain formats in each sub-process, which saves momory and CPU.
	train_g.create_formats_()
	val_g.create_formats_()
	test_g.create_formats_()
	t6 = ttt(t5, "after train_g.create_formats_()")
	tmp = (train_g.in_degrees()==0) & (train_g.out_degrees()==0)
	isolated_nodes = torch.squeeze(torch.nonzero(tmp, as_tuple=False))
	train_g.remove_nodes(isolated_nodes)
	# Pack data
	data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
	       val_nfeat, val_labels, test_nfeat, test_labels
	# get_memory("-----------------------------------------after pack data***************************")
	t7 = ttt(t6, "after pack data")
	run(args, device, data, t6)