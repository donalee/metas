import tensorflow as tf
import numpy as np
import random
from network import ActionNetwork

def train(network_architecture, user_type_pairs, user_adj_dict, item_adj_dict,
			test_data, valid_data, rankingset_data,
			alpha=2.0, learning_rate=0.001, n_updates=5000000, 
			batch_size=200, n_negsamples=3, dropout_prob=0.5, display_step=1000, K=10):

	# Create an ActionNetwork class object
	actionNN = ActionNetwork(network_architecture, 
			alpha=alpha, learning_rate=learning_rate, n_negsamples=n_negsamples)

	n_users = network_architecture['n_users']
	n_items = network_architecture['n_items']
	n_types = network_architecture['n_types']

	for it in range(int(n_updates)):
		batch_anchor, batch_pos, batch_neg = [], [], []
		sampled_anchors = random.sample(user_type_pairs, n_types*10)
		sbatch_size = 0

		# Build a mini-batch for training
		for i in range(len(sampled_anchors)):
			sbatch_anchor, sbatch_pos, sbatch_neg = actionNN.build_minibatch(sampled_anchors[i][0], sampled_anchors[i][1], user_adj_dict, batch_size*2)

			if len(sbatch_anchor) == 0: continue
			batch_anchor.append(sbatch_anchor)
			batch_pos.append(sbatch_pos)
			batch_neg.append(sbatch_neg)

			sbatch_size += len(sbatch_anchor)
			if sbatch_size > batch_size: break

		if len(batch_anchor) == 0: continue
		batch_anchor = np.vstack(batch_anchor)
		batch_pos = np.vstack(batch_pos)
		batch_neg = np.vstack(batch_neg)

		# Optimize the model using the mini-batch of triplets
		actionNN.optimize_network(batch_anchor, batch_pos, batch_neg, dropout_prob)

		# Evaluate the model using the valid dataset
		if it % display_step == 0:
			print("# Triplets: %d" % actionNN.triplet_cnt)
			hits, ndcgs = [[] for i in range(n_types)], [[] for i in range(n_types)]
			for i in range(len(valid_data)):
				hr, ndcg = actionNN.evaluate_triple(valid_data[i], user_adj_dict, item_adj_dict, rankingset_data[i], K)

				typeindex = valid_data[i][2] - 1
				hits[typeindex].append(hr)
				ndcgs[typeindex].append(ndcg)

			TotalHit = sum([sum(hits[i]) for i in range(n_types)])
			TotalNDCG = sum([sum(ndcgs[i]) for i in range(n_types)])

			print("  HR @ %d " % K, end="")
			print(["(Total) %.4f"%(TotalHit/len(valid_data))] + ["(Type-%d) %.4f"%(i+1, sum(hits[i])/len(hits[i])) for i in range(n_types)])
			print("NDCG @ %d " % K, end="")
			print(["(Total) %.4f"%(TotalNDCG/len(valid_data))] + ["(Type-%d) %.4f"%(i+1, sum(ndcgs[i])/len(ndcgs[i])) for i in range(n_types)])
			print("")

	print("Optimization is finished !")
