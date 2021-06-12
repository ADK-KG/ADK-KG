"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Knowledge Graph Environment.
"""

import collections
import os
import pickle

import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import (vector_to_parameters,
											   parameters_to_vector)

from data_utils import load_index
from data_utils import NO_OP_ENTITY_ID, NO_OP_RELATION_ID
from data_utils import DUMMY_ENTITY_ID, DUMMY_RELATION_ID
from data_utils import START_RELATION_ID
import ops as ops
from ops import int_var_cuda, var_cuda

import numpy as np
from collections import defaultdict
from torch.autograd import Variable
import torch.nn.functional as F

from fact_network import get_conve_nn_state_dict, get_conve_kg_state_dict, \
	get_complex_kg_state_dict, get_distmult_kg_state_dict, get_TransE_kg_state_dict


class KnowledgeGraph(nn.Module):
	"""
	The discrete knowledge graph is stored with an adjacency list.
	"""
	def __init__(self, args):
		super(KnowledgeGraph, self).__init__()
		self.entity2id, self.id2entity = {}, {}
		self.relation2id, self.id2relation = {}, {}
		self.type2id, self.id2type = {}, {}
		self.triple2path = {}
		self.entity2typeid = {}
		self.rule = None
		self.adj_list = None
		self.bandwidth = args.bandwidth
		self.args = args
		self.few_shot_relation = None

		self.action_space = None
		self.action_space_buckets = None
		self.unique_r_space = None

		self.train_subjects = None
		self.train_objects = None
		self.dev_subjects = None
		self.dev_objects = None
		self.all_subjects = None
		self.all_objects = None
		self.train_subject_vectors = None
		self.train_object_vectors = None
		self.dev_subject_vectors = None
		self.dev_object_vectors = None
		self.all_subject_vectors = None
		self.all_object_vectors = None

		# pruning state preparation
		#self.prune_method= args.prune_method
		
		#print ("test")
		kg_state_dict_TransE = torch.load(args.TransE_state_dict_path, map_location=('cuda:' + str(args.gpu)))
		kg_state_dict_distmul = torch.load(args.distmult_state_dict_path, map_location=('cuda:' + str(args.gpu)))
		#print ("test")
		#exit(0)
		fn_kg_state_dict_prune = get_TransE_kg_state_dict(kg_state_dict_TransE)
		self.relation_embds_prune = fn_kg_state_dict_prune['relation_embeddings.weight']
		self.entity_embds_prune = fn_kg_state_dict_prune['entity_embeddings.weight']

		#self.entity_type_embds = None

		# print (self.entity_embds_prune[0])
		# exit(0)

		fn_kg_state_dict = get_distmult_kg_state_dict(kg_state_dict_distmul)
		
		# self.relation_embeddings = torch.zeros([289, 100])
		# self.entity_embeddings = torch.zeros([18240, 100])

		#print (self.entity_embeddings.size())

		self.relation_embeddings_true = fn_kg_state_dict['relation_embeddings.weight']
		self.entity_embeddings_true = fn_kg_state_dict['entity_embeddings.weight']
		
		# print (self.relation_embds_prune.size())
		# print (self.entity_embds_prune.size())

		print('** Create {} knowledge graph **'.format(args.model))
		if args.few_shot or args.adaptation:
			self.load_few_shot(args.data_dir)

		self.load_graph_data(args.data_dir)
		self.load_all_answers(args.data_dir)

		# Define NN Modules
		self.entity_dim = args.entity_dim
		self.relation_dim = args.relation_dim
		self.emb_dropout_rate = args.emb_dropout_rate
		self.num_graph_convolution_layers = args.num_graph_convolution_layers
		self.entity_embeddings = None
		self.relation_embeddings = None
		self.entity_img_embeddings = None
		self.relation_img_embeddings = None
		self.EDropout = None
		self.RDropout = None

		#self.entity_type_embds = torch.cuda.FloatTensor(np.array(self.generate_entity_type_embedding(args.data_dir)))
		#self.entity_type_embds = torch.load('entity_type_embds.pt', map_location=('cuda:' + str(args.gpu)))
		#print (self.entity_type_embds.size())
		#self.entity_type_embds = torch.tensor(entity_type_embds, requires_grad=False)
		# print (self.entity_type_embds[-1])
		# print (self.entity_type_embds[-5])
		# exit(0)
		#print (self.entity_type_embds)

		#add for graph neural network - begin
		entity_index_path = args.data_dir + "/entity2id.txt"
		relation_index_path = args.data_dir + "/relation2id.txt"
		self.entity2id_gnn, _ = self.load_index(entity_index_path)
		self.relation2id_gnn, _ = self.load_index(relation_index_path)
		#print (self.entity2id_gnn)

		max_ = 10
		train_graph_path = args.data_dir + "/train.triples"
		graph = np.zeros((len(self.entity2id_gnn), max_, 2))
		e1_rele2 = defaultdict(list)
		e1_degrees = np.zeros(len(self.entity2id_gnn))

		with open(train_graph_path) as f:
			for line in f:
				e1, e2, r = line.strip().split()
				e1_id, e2_id, r_id = self.triple2ids_gnn(e1, e2, r)
				e1_rele2[e1_id].append((r_id, e2_id))
				e1_rele2[e2_id].append((self.relation2id_gnn[r+'_inv'], e1_id))

		for ent in range(len(self.entity2id_gnn)):
			neighbors = e1_rele2[ent]
			if len(neighbors) > max_:
				neighbors = neighbors[:max_]
			# degrees.append(len(neighbors)) 
			e1_degrees[ent] = len(neighbors) + 1# add one for self conn
			for idx, _ in enumerate(neighbors):
				graph[ent, idx, 0] = _[0]
				graph[ent, idx, 1] = _[1]

		#e1_degrees = torch.cuda.FloatTensor(np.zeros(len(self.entity2id_gnn)))
		self.graph = graph
		self.e1_degrees = e1_degrees
		#print (self.e1_degrees)

		self.entity_neigh_agg = nn.Linear(args.entity_dim + args.relation_dim, args.entity_dim)
		self.neigh_att_u = nn.Linear(args.entity_dim, 1)
		self.entity_neigh_self = nn.Linear(args.entity_dim * 2, args.entity_dim)

		#self.entity_neigh_agg_type = nn.Linear(args.entity_dim + args.relation_dim, args.entity_dim)
		#self.neigh_att_u_type = nn.Linear(args.entity_dim, 1)
		#self.entity_neigh_self_type = nn.Linear(args.entity_dim * 2, args.entity_dim)

		self.softmax_m = nn.Softmax(dim=1)

		#add for graph neural network - end
		#self.bn = nn.BatchNorm1d(args.entity_dim)
		
		self.define_modules()
		self.initialize_modules()

	#entity type embedding add
	def generate_entity_type_embedding(self, data_dir):
		entity2typeid = self.entity2typeid
		entity2id, id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
		entity_embds = self.entity_embds_prune.cpu().numpy()

		entity_type_embds = [[0.0] * self.entity_dim] * len(entity2typeid)

		#print ("test")
		entity_type_id = [0] * len(entity2typeid)
		entity_type_dic = {}
		type_id = -1
		entity_id = 0
		for key, value in entity2id.items():
			if len(key.split('_')) > 2:
				type_ = key.split('_')[1]
				if type_ not in entity_type_dic:
					type_id += 1
					entity_type_dic[type_] = type_id
					entity_type_id[entity_id] = type_id
				else:
					entity_type_id[entity_id] = entity_type_dic[type_]

			entity_id += 1

		type_embds = [[0.0] * self.entity_dim] * len(entity_type_dic)
		for i in range(len(entity_type_dic)):
			ave_embed = [0.0] * self.entity_dim
			embed_num = 0 
			for j in range(len(entity_type_id)):
				if entity_type_id[j] == i:
					embed_num += 1
					ave_embed += entity_embds[j]

			type_embds[i] = ave_embed / embed_num

		for k in range(len(entity2typeid)):
			entity_type_embds[k] = type_embds[entity_type_id[k]]

		#print (entity_type_embds)

		return entity_type_embds

	def update_params(self, loss, step_size=0.5, first_order=False):
		# pars = filter(lambda p: p.requires_grad, self.parameters())
		# print (pars)

		grads = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, self.parameters()),
			create_graph=not first_order)

		#print (grads)

		return parameters_to_vector(filter(lambda p: p.requires_grad, self.parameters())) - parameters_to_vector(grads) * step_size
	
	def load_few_shot(self, data_dir):
		self.few_shot_relation = set()
		lines = open(os.path.join(data_dir, 'few_shot.txt')).readlines()
		for line in lines:
			self.few_shot_relation.add(line.strip())

	def load_graph_data(self, data_dir):
		# Load indices
		self.entity2id, self.id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
		print('Sanity check: {} entities loaded'.format(len(self.entity2id)))
		self.type2id, self.id2type = load_index(os.path.join(data_dir, 'type2id.txt'))
		print('Sanity check: {} types loaded'.format(len(self.type2id)))
		with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'rb') as f:
			self.entity2typeid = pickle.load(f)
		self.relation2id, self.id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
		print('Sanity check: {} relations loaded'.format(len(self.relation2id)))
	   
		# Load graph structures
		if self.args.model.startswith('point'): 
			# Base graph structure used for training and test
			adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
			with open(adj_list_path, 'rb') as f:
				self.adj_list = pickle.load(f)
			self.vectorize_action_space(data_dir)

	# def vectorize_action_space(self, data_dir):
	#     """
	#     Pre-process and numericalize the knowledge graph structure.
	#     """
	#     def load_page_rank_scores(input_path):
	#         pgrk_scores = collections.defaultdict(float)
	#         with open(input_path) as f:
	#             for line in f:
	#                 try:
	#                     e, score = line.strip().split(':')
	#                     e_id = self.entity2id[e.strip()]
	#                     score = float(score)
	#                     pgrk_scores[e_id] = score
	#                 except:
	#                     continue
	#         return pgrk_scores
					
	#     # Sanity check
	#     num_facts = 0
	#     out_degrees = collections.defaultdict(int)
	#     for e1 in self.adj_list:
	#         for r in self.adj_list[e1]:
	#             num_facts += len(self.adj_list[e1][r])
	#             out_degrees[e1] += len(self.adj_list[e1][r])
	#     print("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
	#     print('Sanity check: {} facts in knowledge graph'.format(num_facts))

	#     # load page rank scores
	#     page_rank_scores = load_page_rank_scores(os.path.join(data_dir, 'raw.pgrk'))
		
	#     def get_action_space(e1):
	#         action_space = []
	#         if e1 in self.adj_list:
	#             for r in self.adj_list[e1]:
	#                 targets = self.adj_list[e1][r]
	#                 for e2 in targets:
	#                     action_space.append((r, e2))
	#             if len(action_space) + 1 >= self.bandwidth:
	#                 # Base graph pruning
	#                 sorted_action_space = \
	#                     sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
	#                 action_space = sorted_action_space[:self.bandwidth]
	#         action_space.insert(0, (NO_OP_RELATION_ID, e1))
	#         return action_space

	#     def get_unique_r_space(e1):
	#         if e1 in self.adj_list:
	#             return list(self.adj_list[e1].keys())
	#         else:
	#             return []

	#     def vectorize_action_space(action_space_list, action_space_size):
	#         bucket_size = len(action_space_list)
	#         r_space = torch.zeros(bucket_size, action_space_size) + self.dummy_r
	#         e_space = torch.zeros(bucket_size, action_space_size) + self.dummy_e
	#         action_mask = torch.zeros(bucket_size, action_space_size)
	#         for i, action_space in enumerate(action_space_list):
	#             for j, (r, e) in enumerate(action_space):
	#                 r_space[i, j] = r
	#                 e_space[i, j] = e
	#                 action_mask[i, j] = 1
	#         return (int_var_cuda(r_space), int_var_cuda(e_space)), var_cuda(action_mask)

	#     def vectorize_unique_r_space(unique_r_space_list, unique_r_space_size, volatile):
	#         bucket_size = len(unique_r_space_list)
	#         unique_r_space = torch.zeros(bucket_size, unique_r_space_size) + self.dummy_r
	#         for i, u_r_s in enumerate(unique_r_space_list):
	#             for j, r in enumerate(u_r_s):
	#                 unique_r_space[i, j] = r
	#         return int_var_cuda(unique_r_space)

	#     if self.args.use_action_space_bucketing:
	#         """
	#         Store action spaces in buckets.
	#         """
	#         self.action_space_buckets = {}
	#         action_space_buckets_discrete = collections.defaultdict(list)
	#         self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
	#         num_facts_saved_in_action_table = 0
	#         for e1 in range(self.num_entities):
	#             action_space = get_action_space(e1)
	#             key = int(len(action_space) / self.args.bucket_interval) + 1
	#             self.entity2bucketid[e1, 0] = key
	#             self.entity2bucketid[e1, 1] = len(action_space_buckets_discrete[key])
	#             action_space_buckets_discrete[key].append(action_space)
	#             num_facts_saved_in_action_table += len(action_space)
	#         print('Sanity check: {} facts saved in action table'.format(
	#             num_facts_saved_in_action_table - self.num_entities))
	#         for key in action_space_buckets_discrete:
	#             print('Vectorizing action spaces bucket {}...'.format(key))
	#             self.action_space_buckets[key] = vectorize_action_space(
	#                 action_space_buckets_discrete[key], key * self.args.bucket_interval)
	#     else:
	#         action_space_list = []
	#         max_num_actions = 0
	#         for e1 in range(self.num_entities):
	#             action_space = get_action_space(e1)
	#             action_space_list.append(action_space)
	#             if len(action_space) > max_num_actions:
	#                 max_num_actions = len(action_space)
	#         print('Vectorizing action spaces...')
	#         self.action_space = vectorize_action_space(action_space_list, max_num_actions)
			
	#         if self.args.model.startswith('rule'):
	#             unique_r_space_list = []
	#             max_num_unique_rs = 0
	#             for e1 in sorted(self.adj_list.keys()):
	#                 unique_r_space = get_unique_r_space(e1)
	#                 unique_r_space_list.append(unique_r_space)
	#                 if len(unique_r_space) > max_num_unique_rs:
	#                     max_num_unique_rs = len(unique_r_space)
	#             self.unique_r_space = vectorize_unique_r_space(unique_r_space_list, max_num_unique_rs)


	def vectorize_action_space(self, data_dir):
		"""
		Pre-process and numericalize the knowledge graph structure.
		"""
		def load_page_rank_scores(input_path):
			pgrk_scores = collections.defaultdict(float)
			with open(input_path) as f:
				for line in f:
					try:
						e, score = line.strip().split(':')
						e_id = self.entity2id[e.strip()]
						score = float(score)
						pgrk_scores[e_id] = score
					except:
						continue
			return pgrk_scores

		# Sanity check
		num_facts = 0
		out_degrees = collections.defaultdict(int)
		for e1 in self.adj_list:
			for r in self.adj_list[e1]:
				num_facts += len(self.adj_list[e1][r])
				out_degrees[e1] += len(self.adj_list[e1][r])
		print("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
		print('Sanity check: {} facts in knowledge graph'.format(num_facts))

		# load page rank scores
		page_rank_scores = load_page_rank_scores(os.path.join(data_dir, 'raw.pgrk'))
		#embed_dim_prune = self.relation_dim
		#Modified- Mandana
		#def get_action_space(e1):
		def calculate_score(e1,r,e2):
			def dist_mult(E1, R, E2):
				return torch.mm(E1 * R, E2.transpose(1, 0))

			def TransE(E1, R, E2):
				return -torch.sqrt(torch.sum((E1 + R - E2) * (E1 + R - E2), dim=1, keepdim=True))

			r1 = self.relation_embds_prune[r].view(1, 100)
			e1 = self.entity_embds_prune[e1].view(1, 100)
			e2 = self.entity_embds_prune[e2].view(1, 100)

			# R_real=torch.unsqueeze(self.relation_embds[r][0:100], 0)
			# R_img=torch.unsqueeze(self.relation_img_embds[r][0:100], 0)
			# E1_real=torch.unsqueeze(self.entity_embds[e1][100:], 0)
			# E1_img=torch.unsqueeze(self.entity_img_embds[e1][100:], 0)
			# E2_real=torch.unsqueeze(self.entity_embds[e2][100:], 0)
			# E2_img=torch.unsqueeze(self.entity_img_embds[e2][100:], 0)

			# rrr = dist_mult(R_real, E1_real, E2_real)
			# rii = dist_mult(R_real, E1_img, E2_img)
			# iri = dist_mult(R_img, E1_real, E2_img)
			# iir = dist_mult(R_img, E1_img, E2_real)
			# S = rrr + rii + iri - iir

			S = TransE(e1, r1, e2)
			S = F.sigmoid(S)

			return S

		def get_action_space(e1):
			prunning_criteria='page_rank'
			#prunning_criteria='distmul'
			action_space = []
			score_dic=defaultdict(lambda: defaultdict(int))
			if e1 in self.adj_list:
				for r in self.adj_list[e1]:
					targets = self.adj_list[e1][r]
					for e2 in targets:
						score=calculate_score(e1,r,e2)
						action_space.append((r, e2))
						score_dic[r][e2]=score
				if len(action_space) + 1 >= self.bandwidth:
					# Base graph pruning
					if prunning_criteria != 'page_rank':
						sorted_action_space = \
							sorted(action_space, key=lambda x: score_dic[x[0]][x[1]], reverse=True)
					else:
						sorted_action_space = \
							sorted(action_space, key=lambda x: page_rank_scores[x[1]],reverse=True)
					action_space = sorted_action_space[:self.bandwidth]
			action_space.insert(0, (NO_OP_RELATION_ID, e1))
			return action_space

		def get_unique_r_space(e1):
			if e1 in self.adj_list:
				return list(self.adj_list[e1].keys())
			else:
				return []

		def vectorize_action_space(action_space_list, action_space_size):
			bucket_size = len(action_space_list)
			r_space = torch.zeros(bucket_size, action_space_size) + self.dummy_r
			e_space = torch.zeros(bucket_size, action_space_size) + self.dummy_e
			action_mask = torch.zeros(bucket_size, action_space_size)
			for i, action_space in enumerate(action_space_list):
				for j, (r, e) in enumerate(action_space):
					r_space[i, j] = r
					e_space[i, j] = e
					action_mask[i, j] = 1
			return (int_var_cuda(r_space), int_var_cuda(e_space)), var_cuda(action_mask)

		def vectorize_unique_r_space(unique_r_space_list, unique_r_space_size, volatile):
			bucket_size = len(unique_r_space_list)
			unique_r_space = torch.zeros(bucket_size, unique_r_space_size) + self.dummy_r
			for i, u_r_s in enumerate(unique_r_space_list):
				for j, r in enumerate(u_r_s):
					unique_r_space[i, j] = r
			return int_var_cuda(unique_r_space)

		if self.args.use_action_space_bucketing:
			"""
			Store action spaces in buckets.
			"""
			self.action_space_buckets = {}
			action_space_buckets_discrete = collections.defaultdict(list)
			self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
			num_facts_saved_in_action_table = 0
			for e1 in range(self.num_entities):
				action_space = get_action_space(e1)
				key = int(len(action_space) / self.args.bucket_interval) + 1
				self.entity2bucketid[e1, 0] = key
				self.entity2bucketid[e1, 1] = len(action_space_buckets_discrete[key])
				action_space_buckets_discrete[key].append(action_space)
				num_facts_saved_in_action_table += len(action_space)
			print('Sanity check: {} facts saved in action table'.format(
				num_facts_saved_in_action_table - self.num_entities))
			for key in action_space_buckets_discrete:
				print('Vectorizing action spaces bucket {}...'.format(key))
				self.action_space_buckets[key] = vectorize_action_space(
					action_space_buckets_discrete[key], key * self.args.bucket_interval)
		else:
			action_space_list = []
			max_num_actions = 0
			for e1 in range(self.num_entities):
				action_space = get_action_space(e1)
				action_space_list.append(action_space)
				if len(action_space) > max_num_actions:
					max_num_actions = len(action_space)
			print('Vectorizing action spaces...')
			self.action_space = vectorize_action_space(action_space_list, max_num_actions)

			if self.args.model.startswith('rule'):
				unique_r_space_list = []
				max_num_unique_rs = 0
				for e1 in sorted(self.adj_list.keys()):
					unique_r_space = get_unique_r_space(e1)
					unique_r_space_list.append(unique_r_space)
					if len(unique_r_space) > max_num_unique_rs:
						max_num_unique_rs = len(unique_r_space)
				self.unique_r_space = vectorize_unique_r_space(unique_r_space_list, max_num_unique_rs)

	def load_all_answers(self, data_dir, add_reversed_edges=False):
		def add_subject(e1, e2, r, d):
			if not e2 in d:
				d[e2] = {}
			if not r in d[e2]:
				d[e2][r] = set()
			d[e2][r].add(e1)

		def add_object(e1, e2, r, d):
			if not e1 in d:
				d[e1] = {}
			if not r in d[e1]:
				d[e1][r] = set()
			d[e1][r].add(e2)

		# store subjects for all (rel, object) queries and
		# objects for all (subject, rel) queries
		train_subjects, train_objects = {}, {}
		dev_subjects, dev_objects = {}, {}
		all_subjects, all_objects = {}, {}
		# include dummy examples
		add_subject(self.dummy_e, self.dummy_e, self.dummy_r, train_subjects)
		add_subject(self.dummy_e, self.dummy_e, self.dummy_r, dev_subjects)
		add_subject(self.dummy_e, self.dummy_e, self.dummy_r, all_subjects)
		add_object(self.dummy_e, self.dummy_e, self.dummy_r, train_objects)
		add_object(self.dummy_e, self.dummy_e, self.dummy_r, dev_objects)
		add_object(self.dummy_e, self.dummy_e, self.dummy_r, all_objects)
		for file_name in ['raw.kb', 'train.triples', 'dev.triples', 'test.triples']:
			if 'NELL' in self.args.data_dir and self.args.test and file_name == 'train.triples':
				continue
			with open(os.path.join(data_dir, file_name)) as f:
				for line in f:
					e1, e2, r = line.strip().split()
					#print (line)
					e1, e2, r = self.triple2ids((e1, e2, r))
					if file_name in ['raw.kb', 'train.triples']:
						add_subject(e1, e2, r, train_subjects)
						add_object(e1, e2, r, train_objects)
						if add_reversed_edges:
							add_subject(e2, e1, self.get_inv_relation_id(r), train_subjects)
							add_object(e2, e1, self.get_inv_relation_id(r), train_objects)
					if file_name in ['raw.kb', 'train.triples', 'dev.triples']:
						add_subject(e1, e2, r, dev_subjects)
						add_object(e1, e2, r, dev_objects)
						if add_reversed_edges:
							add_subject(e2, e1, self.get_inv_relation_id(r), dev_subjects)
							add_object(e2, e1, self.get_inv_relation_id(r), dev_objects)
					add_subject(e1, e2, r, all_subjects)
					add_object(e1, e2, r, all_objects)
					if add_reversed_edges:
						add_subject(e2, e1, self.get_inv_relation_id(r), all_subjects)
						add_object(e2, e1, self.get_inv_relation_id(r), all_objects)
		self.train_subjects = train_subjects
		self.train_objects = train_objects
		self.dev_subjects = dev_subjects
		self.dev_objects = dev_objects
		self.all_subjects = all_subjects
		self.all_objects = all_objects
	   
		# change the answer set into a variable
		def answers_to_var(d_l):
			d_v = collections.defaultdict(collections.defaultdict)
			for x in d_l:
				for y in d_l[x]:
					v = torch.LongTensor(list(d_l[x][y])).unsqueeze(1)
					d_v[x][y] = int_var_cuda(v)
			return d_v
		
		self.train_subject_vectors = answers_to_var(train_subjects)
		self.train_object_vectors = answers_to_var(train_objects)
		self.dev_subject_vectors = answers_to_var(dev_subjects)
		self.dev_object_vectors = answers_to_var(dev_objects)
		self.all_subject_vectors = answers_to_var(all_subjects)
		self.all_object_vectors = answers_to_var(all_objects)

	def load_fuzzy_facts(self):
		# extend current adjacency list with fuzzy facts
		dev_path = os.path.join(self.args.data_dir, 'dev.triples.txt')
		test_path = os.path.join(self.args.data_dir, 'test.triples.txt')
		with open(dev_path) as f:
			dev_triples = [l.strip() for l in f.readlines()]
		with open(test_path) as f:
			test_triples = [l.strip() for l in f.readlines()]
		removed_triples = set(dev_triples + test_triples)
		theta = 0.5
		fuzzy_fact_path = os.path.join(self.args.data_dir, 'train.fuzzy.triples')
		count = 0
		with open(fuzzy_fact_path) as f:
			for line in f:
				e1, e2, r, score = line.strip().split()
				score = float(score)
				if score < theta:
					continue
				print(line)
				if '{}\t{}\t{}'.format(e1, e2, r) in removed_triples:
					continue
				e1_id = self.entity2id[e1]
				e2_id = self.entity2id[e2]
				r_id = self.relation2id[r]
				if not r_id in self.adj_list[e1_id]:
					self.adj_list[e1_id][r_id] = set()
				if not e2_id in self.adj_list[e1_id][r_id]:
					self.adj_list[e1_id][r_id].add(e2_id)
					count += 1
					if count > 0 and count % 1000 == 0:
						print('{} fuzzy facts added'.format(count))

		self.vectorize_action_space(self.args.data_dir)

	def get_inv_relation_id(self, r_id):
		return r_id + 1

	def get_all_entity_embeddings(self):
		return self.EDropout(self.entity_embeddings.weight)

	def get_entity_embeddings(self, e):
		return self.EDropout(self.entity_embeddings(e))

	def get_entity_type_embeddings(self, e):
		return self.EDropout(self.entity_type_embds[e])

	def get_entity_embeddings_pretrain(self, e):
		return self.EDropout(self.entity_embds_prune[e])

	def get_relation_embeddings_pretrain(self, r):
		return self.RDropout(self.relation_embds_prune[r])

	def get_all_relation_embeddings(self):
		return self.RDropout(self.relation_embeddings.weight)

	def get_relation_embeddings(self, r):
		return self.RDropout(self.relation_embeddings(r))

	def get_all_entity_img_embeddings(self):
		return self.EDropout(self.entity_img_embeddings.weight)

	def get_entity_img_embeddings(self, e):
		return self.EDropout(self.entity_img_embeddings(e))

	def get_relation_img_embeddings(self, r):
		return self.RDropout(self.relation_img_embeddings(r))

	def virtual_step(self, e_set, r):
		"""
		Given a set of entities (e_set), find the set of entities (e_set_out) which has at least one incoming edge
		labeled r and the source entity is in e_set.
		"""
		batch_size = len(e_set)
		e_set_1D = e_set.view(-1)
		r_space = self.action_space[0][0][e_set_1D]
		e_space = self.action_space[0][1][e_set_1D]
		e_space = (r_space.view(batch_size, -1) == r.unsqueeze(1)).long() * e_space.view(batch_size, -1)
		e_set_out = []
		for i in range(len(e_space)):
			e_set_out_b = var_cuda(unique(e_space[i].data))
			e_set_out.append(e_set_out_b.unsqueeze(0))
		e_set_out = ops.pad_and_cat(e_set_out, padding_value=self.dummy_e)
		return e_set_out

	def id2triples(self, triple):
		e1, e2, r = triple
		return self.id2entity[e1], self.id2entity[e2], self.id2relation[r]

	def triple2ids(self, triple):
		e1, e2, r = triple
		return self.entity2id[e1], self.entity2id[e2], self.relation2id[r]

	def define_modules(self):
		if not self.args.relation_only:
			self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
			#self.entity_embeddings = nn.Embedding(18240, self.entity_dim)
			#self.entity_embeddings_true = nn.Embedding(self.num_entities, self.entity_dim)
			if self.args.model == 'complex':
				self.entity_img_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
			self.EDropout = nn.Dropout(self.emb_dropout_rate)
		self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
		#self.relation_embeddings_true = nn.Embedding(self.num_relations, self.relation_dim)
		if self.args.model == 'complex':
			self.relation_img_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
		self.RDropout = nn.Dropout(self.emb_dropout_rate)

	def initialize_modules(self):
		if not self.args.relation_only:
			nn.init.xavier_normal_(self.entity_embeddings.weight)
		nn.init.xavier_normal_(self.relation_embeddings.weight)
		
		nn.init.xavier_normal_(self.entity_neigh_agg.weight)
		nn.init.xavier_normal_(self.entity_neigh_self.weight)
		nn.init.xavier_normal_(self.neigh_att_u.weight)

		#nn.init.xavier_normal_(self.entity_neigh_agg_type.weight)
		#nn.init.xavier_normal_(self.entity_neigh_self_type.weight)
		#nn.init.xavier_normal_(self.neigh_att_u_type.weight)

		self.entity_embeddings.weight.requires_grad = False
		self.relation_embeddings.weight.requires_grad = False
		#self.entity_type_embds.requires_grad = False

		#torch.save(self.entity_type_embds, 'entity_type_embds.pt')

	@property
	def num_entities(self):
		return len(self.entity2id)

	@property
	def num_relations(self):
		return len(self.relation2id)

	@property
	def self_edge(self):
		return NO_OP_RELATION_ID

	@property
	def self_e(self):
		return NO_OP_ENTITY_ID        

	@property
	def dummy_r(self):
		return DUMMY_RELATION_ID

	@property
	def dummy_e(self):
		return DUMMY_ENTITY_ID

	@property
	def dummy_start_r(self):
		return START_RELATION_ID

	def load_index(self, input_path):
		index, rev_index = {}, {}
		with open(input_path) as f:
			for i, line in enumerate(f.readlines()):
				v, _ = line.strip().split()
				index[v] = i
				rev_index[i] = v
		return index, rev_index

	def triple2ids_gnn(self, e1, e2, r):
		return self.entity2id_gnn[e1], self.entity2id_gnn[e2], self.relation2id_gnn[r]

	def neighbor_encoder(self, e):
		d_value = list(e.size())

		d_num = len(list(d_value))

		e_temp = e.view(-1)

		num_neigh = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in e_temp])).unsqueeze(1).cuda()
		graph = Variable(torch.LongTensor(np.stack([self.graph[_,:,:] for _ in e_temp], axis=0))).cuda()
		
		relations = graph[:,:,0].squeeze(-1)
		entities = graph[:,:,1].squeeze(-1)

		#rel_embeds = self.get_relation_embeddings_pretrain(relations)

		rel_embeds = self.get_relation_embeddings_pretrain(relations)

		ent_embeds = self.gnn(entities)

		#ent_embeds = self.get_entity_embeddings(entities)

		# print (ent_embeds.size())
		# exit(0)

		#neighbor encoder
		concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
		out_0 = self.entity_neigh_agg(concat_embeds)
		out = torch.sum(out_0, dim=1) # (batch, embed_dim)
		out = out / num_neigh
		out = out.tanh()

		#out = self.bn(out)

		#mean pooling
		# out = torch.sum(ent_embeds, dim=1)
		# out = out / num_neigh
		# out = out.tanh()

		# print (out.size())
		# exit (0)

		if d_num == 1:
			out = out.view(d_value[0], -1)
		elif d_num == 2:
			out = out.view(d_value[0], d_value[1], -1)

		return out

	def gnn(self, e):
		size = list(e.size())
		#print (size)
		e_temp = e.view(-1)
		num_neigh = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in e_temp])).unsqueeze(1).cuda()
		graph = Variable(torch.LongTensor(np.stack([self.graph[_,:,:] for _ in e_temp], axis=0))).cuda()

		relations = graph[:,:,0].squeeze(-1)
		entities = graph[:,:,1].squeeze(-1)

		rel_embeds = self.get_relation_embeddings_pretrain(relations)
		ent_embeds = self.get_entity_embeddings_pretrain(entities)

		#neighbor encoder
		concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
		out_0 = self.entity_neigh_agg(concat_embeds)
		out = torch.sum(out_0, dim=1) # (batch, embed_dim)
		out = out / num_neigh
		out = out.tanh()

		out = out.view(size[0], size[1], -1)

		#print (out.size())
		#exit(0)

		return out

	def het_gnn(self, e):
		#print (e.size())
		#print (self.e1_degrees)
		d_value = list(e.size())
		d_num = len(list(d_value))

		e_temp = e.view(-1)
		num_neigh = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in e_temp])).unsqueeze(1).cuda()
		graph = Variable(torch.LongTensor(np.stack([self.graph[_,:,:] for _ in e_temp], axis=0))).cuda()

		relations = graph[:,:,0].squeeze(-1)
		entities = graph[:,:,1].squeeze(-1)

		rel_embeds = self.get_relation_embeddings_pretrain(relations)
		ent_embeds = self.get_entity_embeddings_pretrain(entities)

		concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)

		#print (concat_embeds.size())

		out_0 = self.entity_neigh_agg(concat_embeds).tanh()

		# out = torch.sum(out_0, dim=1) # (batch, embed_dim)
		# out = out / num_neigh
		# out = out.tanh()

		att_w = self.neigh_att_u(out_0).tanh()
		att_w = self.softmax_m(att_w)
		#print (att_w.size())
		#exit(0)

		att_w = att_w.view(concat_embeds.size()[0], 1, 10)
		out = torch.bmm(att_w, concat_embeds).view(concat_embeds.size()[0], self.entity_dim * 2)
		out = self.entity_neigh_self(out).tanh()

		# out = out.tanh()

		# print (out.size())
		# exit(0)

		# self_embed = self.get_entity_embeddings(e_temp)
		# out = torch.cat((out, self_embed), dim=-1)
		# out = self.entity_neigh_self(out).tanh()
			
		if d_num == 1:
			out = out.view(d_value[0], -1)
		elif d_num == 2:
			out = out.view(d_value[0], d_value[1], -1)

		#return self_embed

		return out

	def het_gnn_with_type(self, e):
		#print (e.size())
		#print (self.e1_degrees)
		d_value = list(e.size())
		d_num = len(list(d_value))

		e_temp = e.view(-1)
		num_neigh = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in e_temp])).unsqueeze(1).cuda()
		graph = Variable(torch.LongTensor(np.stack([self.graph[_,:,:] for _ in e_temp], axis=0))).cuda()

		relations = graph[:,:,0].squeeze(-1)
		entities = graph[:,:,1].squeeze(-1)

		rel_embeds = self.get_relation_embeddings_pretrain(relations)
		ent_embeds = self.get_entity_embeddings_pretrain(entities)

		#print (ent_embeds.size())

		concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)

		#print (concat_embeds.size())

		out_0 = self.entity_neigh_agg(concat_embeds).tanh()

		# out = torch.sum(out_0, dim=1) # (batch, embed_dim)
		# out = out / num_neigh
		# out = out.tanh()

		att_w = self.neigh_att_u(out_0).tanh()
		att_w = self.softmax_m(att_w)
		#print (att_w.size())
		#exit(0)
		
		#print (ent_embeds)

		att_w = att_w.view(concat_embeds.size()[0], 1, 10)
		out = torch.bmm(att_w, concat_embeds).view(concat_embeds.size()[0], self.entity_dim * 2)
		out = self.entity_neigh_self(out).tanh()


		#type embedding add
		ent_embeds_type = self.get_entity_type_embeddings(entities)
		#print (ent_embeds_type.size())
		#print (ent_embeds_type)

		concat_embeds_type = torch.cat((rel_embeds, ent_embeds_type), dim=-1)

		out_0_type = self.entity_neigh_agg_type(concat_embeds_type).tanh()

		att_w_type = self.neigh_att_u_type(out_0_type).tanh()
		att_w_type = self.softmax_m(att_w_type)

		att_w_type = att_w.view(concat_embeds_type.size()[0], 1, 10)
		out_type = torch.bmm(att_w_type, concat_embeds_type).view(concat_embeds_type.size()[0], self.entity_dim * 2)
		out_type = self.entity_neigh_self_type(out_type).tanh()

		# print (out.size())
		# print (out_type.size())
		# exit(0)

		out = (out + out_type) / 2

		# out = out.tanh()

		# print (out.size())
		# exit(0)

		# self_embed = self.get_entity_embeddings(e_temp)
		# out = torch.cat((out, self_embed), dim=-1)
		# out = self.entity_neigh_self(out).tanh()
			
		if d_num == 1:
			out = out.view(d_value[0], -1)
		elif d_num == 2:
			out = out.view(d_value[0], d_value[1], -1)

		#return self_embed

		#print (out)

		return out



