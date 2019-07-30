import numpy as np
import pandas as pd
import math

def read_file(filename):

	train_filename = filename + "-train.csv"
	test_filename = filename + "-test.csv"
	valid_filename = filename + "-valid.csv"
	rankingset_filename = filename + "-ranking-set.csv"

	train_data = pd.read_csv(train_filename, header=None)
	test_data = pd.read_csv(test_filename, header=None)
	valid_data = pd.read_csv(valid_filename, header=None)
	rankingset_data = pd.read_csv(rankingset_filename, header=None)

	train_data = train_data.values
	test_data = test_data.values
	valid_data = valid_data.values
	rankingset_data = rankingset_data.values

	users, items, types = set(), set(), set()
	user_adj_dict, item_adj_dict = dict(), dict()

	for i in range(len(train_data)):
		userkey, itemkey, typekey = train_data[i]

		if typekey not in user_adj_dict:
			user_adj_dict[typekey] = dict()
			item_adj_dict[typekey] = dict()

		if userkey not in user_adj_dict[typekey]:
			user_adj_dict[typekey][userkey] = set()

		if itemkey not in item_adj_dict[typekey]:
			item_adj_dict[typekey][itemkey] = set()

		users.add(userkey)
		items.add(itemkey)
		types.add(typekey)
		
		user_adj_dict[typekey][userkey].add(train_data[i][1])
		item_adj_dict[typekey][itemkey].add(train_data[i][0])

	n_users, n_items, n_types = len(users), len(items), len(types)

	user_type_pairs = []
	for typekey in range(1, n_types+1):
		for userkey in user_adj_dict[typekey]:
			user_type_pairs.append([userkey, typekey])

	return [n_users, n_items, n_types], user_type_pairs, user_adj_dict, item_adj_dict, test_data, valid_data, rankingset_data

def getHitRatio(ranklist, target):
	for item in ranklist:
		if item == target:
			return 1
	return 0

def getNDCG(ranklist, target):
	for i, item in enumerate(ranklist):
		if item == target:
			return math.log(2) / math.log(i+2)
	return 0
