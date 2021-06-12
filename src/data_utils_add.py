import numpy as np
import re
import os
import random
from csv import reader
import collections
import pickle

START_RELATION = 'START_RELATION'
NO_OP_RELATION = 'NO_OP_RELATION'
NO_OP_ENTITY = 'NO_OP_ENTITY'
DUMMY_RELATION = 'DUMMY_RELATION'
DUMMY_ENTITY = 'DUMMY_ENTITY'

DUMMY_RELATION_ID = 0
START_RELATION_ID = 1
NO_OP_RELATION_ID = 2
DUMMY_ENTITY_ID = 0
NO_OP_ENTITY_ID = 1

data_dir = "../data/FB-new/"


def data_process():
	r_dic = {}
	with open(os.path.join(data_dir, 'all.triples')) as f:
		for line in f:
			e1, e2, r = line.strip().split('\t')
			if r not in r_dic:
				r_dic[r] = 1
			else:
				r_dic[r] += 1

	select_r_dic = {}
	for key, value in r_dic.items():
		if value >= 50 and value <= 500:
			select_r_dic[key] = 1

	#print len(select_r_dic)

	few_shot_triples_f = open(data_dir + "few-shot-all.triples.txt", "w")
	with open(os.path.join(data_dir, 'all.triples')) as f:
		for line in f:
			e1, e2, r = line.strip().split('\t')
			if r in select_r_dic:
				few_shot_triples_f.write(e1 + "\t" + e2 + "\t" + r + "\n") 
	few_shot_triples_f.close()

	#new id
	r_dic_new = {}
	entity_dic_new = {}
	with open(os.path.join(data_dir, 'few-shot-all.triples.txt')) as f:
		for line in f:
			e1, e2, r = line.strip().split('\t')
			if r not in r_dic_new:
				r_dic_new[r] = 1
			else:
				r_dic_new[r] += 1

			if e1 not in entity_dic_new:
				entity_dic_new[e1] = 1
			else:
				entity_dic_new[e1] += 1

			if e2 not in entity_dic_new:
				entity_dic_new[e2] = 1
			else:
				entity_dic_new[e2] += 1

	r_dic_new_sort = sorted(r_dic_new.items(), key=lambda kv: kv[1], reverse=True)
	entity_dic_new_sort = sorted(entity_dic_new.items(), key=lambda kv: kv[1], reverse=True)

	select_test_r_f = open(data_dir + "few_shot.txt", "w")
	select_test_r_id = random.sample(range(0, len(r_dic_new)), 20)
	#print (select_test_r_id)
	select_test_r_dic = {}
	select_test_r_dic_2 = {}
	for i in range(len(select_test_r_id)):
		# select_test_r_dic[r_dic_new_sort[select_test_r_id[i]][0]] = 1
		# select_test_r_f.write(r_dic_new_sort[select_test_r_id[i]][0] + "\n") 
		select_test_r_dic[r_dic_new_sort[i][0]] = 1
		select_test_r_dic_2[r_dic_new_sort[i][0]] = 1
		select_test_r_f.write(r_dic_new_sort[i][0] + "\n") 
	select_test_r_f.close()

	few_shot_size = 20
	visit_entity = {}
	train_triples_f = open(data_dir + "train.triples.txt", "w")
	with open(os.path.join(data_dir, 'few-shot-all.triples.txt')) as f:
		for line in f:
			e1, e2, r = line.strip().split('\t')
			if r in select_test_r_dic:
				if select_test_r_dic[r] <= few_shot_size:
					train_triples_f.write(e1 + "\t" + e2 + "\t" + r + "\n") 	
					select_test_r_dic[r] += 1
					visit_entity[e1] = 1
					visit_entity[e2] = 1		
			else:
				train_triples_f.write(e1 + "\t" + e2 + "\t" + r + "\n") 
				visit_entity[e1] = 1
				visit_entity[e2] = 1
	train_triples_f.close()
	
	test_triples_f = open(data_dir + "test.triples.txt", "w")
	with open(os.path.join(data_dir, 'few-shot-all.triples.txt')) as f:
		for line in f:
			e1, e2, r = line.strip().split('\t')
			if r in select_test_r_dic_2:
				if select_test_r_dic_2[r] > few_shot_size:
					if e1 in visit_entity and e2 in visit_entity:
						test_triples_f.write(e1 + "\t" + e2 + "\t" + r + "\n")
				else:
					select_test_r_dic_2[r] += 1
	test_triples_f.close()



def prepare_kb_envrioment(test_mode, add_reverse_relations=True):
	"""
	Process KB data which was saved as a set of triples.
		(a) Remove train and test triples from the KB envrionment.
		(b) Add reverse triples on demand.
		(c) Index unique entities and relations appeared in the KB.

	:param raw_kb_path: Path to the raw KB triples.
	:param train_path: Path to the train set KB triples.
	:param dev_path: Path to the dev set KB triples.
	:param test_path: Path to the test set KB triples.
	:param add_reverse_relations: If set, add reverse triples to the KB environment.
	"""
	#data_dir = os.path.dirname(raw_kb_path)

	def get_type(e_name):
		if e_name == DUMMY_ENTITY:
			return DUMMY_ENTITY
		if 'nell-995' in data_dir.lower():
			if '_' in e_name:
				return e_name.split('_')[1]
			else:
				return 'numerical'
		else:
			return 'entity'

	def hist_to_vocab(_dict):
		return sorted(sorted(_dict.items(), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)

	# Create entity and relation indices
	entity_hist = collections.defaultdict(int)
	relation_hist = collections.defaultdict(int)
	type_hist = collections.defaultdict(int)
	with open(os.path.join(data_dir, 'raw.kb.txt')) as f:
	#with open(os.path.join(raw_kb_path)) as f:
		raw_kb_triples = [l.strip() for l in f.readlines()]
	with open(os.path.join(data_dir, 'train.triples.txt')) as f:
	#with open(os.path.join(train_path)) as f:
		train_triples = [l.strip() for l in f.readlines()]
	with open(os.path.join(data_dir, 'test.triples.txt')) as f:
	#with open(os.path.join(dev_path)) as f:
		dev_triples = [l.strip() for l in f.readlines()]
	with open(os.path.join(data_dir, 'test.triples.txt')) as f:
	#with open(os.path.join(test_path)) as f:
		test_triples = [l.strip() for l in f.readlines()]

	if test_mode:
		keep_triples = train_triples + dev_triples
		removed_triples = test_triples
	else:
		keep_triples = train_triples
		removed_triples = dev_triples + test_triples

	# Index entities and relations
	for line in set(raw_kb_triples + keep_triples + removed_triples):
		e1, e2, r = line.strip().split()
		entity_hist[e1] += 1
		entity_hist[e2] += 1
		if 'nell-995' in data_dir.lower():
			t1 = e1.split('_')[1] if '_' in e1 else 'numerical'
			t2 = e2.split('_')[1] if '_' in e2 else 'numerical'
		else:
			t1 = get_type(e1)
			t2 = get_type(e2)
		type_hist[t1] += 1
		type_hist[t2] += 1
		relation_hist[r] += 1
		if add_reverse_relations:
			inv_r = r + '_inv'
			relation_hist[inv_r] += 1
	# Save the entity and relation indices sorted by decreasing frequency
	with open(os.path.join(data_dir, 'entity2id.txt'), 'w') as o_f:
		o_f.write('{}\t{}\n'.format(DUMMY_ENTITY, DUMMY_ENTITY_ID))
		o_f.write('{}\t{}\n'.format(NO_OP_ENTITY, NO_OP_ENTITY_ID))
		for e, freq in hist_to_vocab(entity_hist):
			o_f.write('{}\t{}\n'.format(e, freq))
	with open(os.path.join(data_dir, 'relation2id.txt'), 'w') as o_f:
		o_f.write('{}\t{}\n'.format(DUMMY_RELATION, DUMMY_RELATION_ID))
		o_f.write('{}\t{}\n'.format(START_RELATION, START_RELATION_ID))
		o_f.write('{}\t{}\n'.format(NO_OP_RELATION, NO_OP_RELATION_ID))
		for r, freq in hist_to_vocab(relation_hist):
			o_f.write('{}\t{}\n'.format(r, freq))
	with open(os.path.join(data_dir, 'type2id.txt'), 'w') as o_f:
		for t, freq in hist_to_vocab(type_hist):
			o_f.write('{}\t{}\n'.format(t, freq))
	print('{} entities indexed'.format(len(entity_hist)))
	print('{} relations indexed'.format(len(relation_hist)))
	print('{} types indexed'.format(len(type_hist)))
	entity2id, id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
	relation2id, id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
	type2id, id2type = load_index(os.path.join(data_dir, 'type2id.txt'))

	removed_triples = set(removed_triples)
	adj_list = collections.defaultdict(collections.defaultdict)
	entity2typeid = [0 for i in range(len(entity2id))]
	num_facts = 0
	for line in set(raw_kb_triples + keep_triples):
		e1, e2, r = line.strip().split()
		triple_signature = '{}\t{}\t{}'.format(e1, e2, r)
		e1_id = entity2id[e1]
		e2_id = entity2id[e2]
		t1 = get_type(e1)
		t2 = get_type(e2)
		t1_id = type2id[t1]
		t2_id = type2id[t2]
		entity2typeid[e1_id] = t1_id
		entity2typeid[e2_id] = t2_id
		if not triple_signature in removed_triples:
			r_id = relation2id[r]
			if not r_id in adj_list[e1_id]:
				adj_list[e1_id][r_id] = set()
			if e2_id in adj_list[e1_id][r_id]:
				print('Duplicate fact: {} ({}, {}, {})!'.format(
					line.strip(), id2entity[e1_id], id2relation[r_id], id2entity[e2_id]))
			adj_list[e1_id][r_id].add(e2_id)
			num_facts += 1
			if add_reverse_relations:
				inv_r = r + '_inv'
				inv_r_id = relation2id[inv_r]
				if not inv_r_id in adj_list[e2_id]:
					adj_list[e2_id][inv_r_id] = set([])
				if e1_id in adj_list[e2_id][inv_r_id]:
					print('Duplicate fact: {} ({}, {}, {})!'.format(
						line.strip(), id2entity[e2_id], id2relation[inv_r_id], id2entity[e1_id]))
				adj_list[e2_id][inv_r_id].add(e1_id)
				num_facts += 1
	print('{} facts processed'.format(num_facts))
	# Save adjacency list
	adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
	with open(adj_list_path, 'wb') as o_f:
		pickle.dump(dict(adj_list), o_f)
	with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'wb') as o_f:
		pickle.dump(entity2typeid, o_f)

	#compute pagerank score
	entity_pgrk = {}
	entity_deg = {}
	for key in adj_list:
		entity_pgrk[key] = 0.001

	for entity_iter in adj_list:
		deg_temp = 0
		for rel_iter in adj_list[entity_iter]:
			for neigh_iter in adj_list[entity_iter][rel_iter]:
				deg_temp += 1
		entity_deg[entity_iter] = deg_temp

	#print (len(entity_pgrk))

	iter_max = 1000
	#thredshold = 1e-5
	for t in range(iter_max):
		for entity_iter in entity_pgrk:
			score_temp = 0.0
			for rel_iter in adj_list[entity_iter]:
				for neigh_iter in adj_list[entity_iter][rel_iter]:
					score_temp += entity_pgrk[neigh_iter] / entity_deg[neigh_iter]
			entity_pgrk[entity_iter] = score_temp

	# temp = 0
	# for entity_iter_2 in entity_pgrk:
	# 	if temp == 0:
	# 		print (entity_pgrk[entity_iter_2])
	# 	temp += 1

	pgrk_f = open(data_dir + "raw.pgrk", "w")
	for entity_i in entity_pgrk:
		pgrk_f.write(str(id2entity[entity_i]) + ":" + str(entity_pgrk[entity_i]) + "\n") 
	pgrk_f.close()



def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, _ = line.strip().split()
            index[v] = i
            rev_index[i] = v
    return index, rev_index



#data_process()


#prepare_kb_envrioment(0, add_reverse_relations = True)






