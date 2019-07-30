import tensorflow as tf
import pandas as pd
import numpy as np
import math, heapq, random, time
from utils import getHitRatio, getNDCG


class ActionNetwork(object):
	"""
	This class includes MLP networks to obtain the action space
	(i.e., embedding vectors of (1) behavior models and (2) actions)
	"""
	def __init__(self, network_architecture, alpha=2.0, learning_rate=0.001, n_negsamples=3):
		self.net_arch = network_architecture
		self.alpha = alpha
		self.learning_rate = learning_rate
		self.n_negsamples = n_negsamples
		self.triplet_cnt = 0

		# Define tensorflow graph inputs
		self.anchor_x = tf.placeholder(tf.int32, [None, 2])
		self.pos_x = tf.placeholder(tf.int32, [None, 3])
		self.neg_x = tf.placeholder(tf.int32, [None, 3])
		self.dropout_x = tf.placeholder(tf.float32)

		# Create two mapping functions f and g
		self._create_network()

		# Define the loss function
		self._create_loss_optimizer()

		# Initialize the variables
		init = tf.global_variables_initializer()

		# Launch the session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		self.sess.run(init)

	def _create_network(self):
		anchor_weights = self._initialize_weights(**self.net_arch, mlp_name="mlp_g", embed_reuse=False, mlp_reuse=False)
		self.anchor = self._MLP_g(self.anchor_x, self.dropout_x,
					anchor_weights['embeddings'], anchor_weights['weights'], anchor_weights['biases'])

		pos_weights = self._initialize_weights(**self.net_arch, mlp_name="mlp_f", embed_reuse=True, mlp_reuse=False)
		self.pos = self._MLP_f(self.pos_x, self.dropout_x,
					pos_weights['embeddings'], pos_weights['weights'], pos_weights['biases'])

		neg_weights = self._initialize_weights(**self.net_arch, mlp_name="mlp_f", embed_reuse=True, mlp_reuse=True)
		self.neg = self._MLP_f(self.neg_x, self.dropout_x,
					neg_weights['embeddings'], neg_weights['weights'], neg_weights['biases'])

	def _initialize_weights(self, n_users, n_items, n_types, n_embdims, n_layers, n_hiddens, mlp_name, embed_reuse=False, mlp_reuse=False):
		all_weights = dict()

		if mlp_name == "mlp_f":
			n_hidden_0 = n_embdims*3
		else:
			n_hidden_0 = n_embdims*2

		n_hidden_1, n_hidden_2, n_hidden_3 = [int(n_hidden) for n_hidden in n_hiddens.split()]

		with tf.variable_scope("embedding", reuse=embed_reuse):
			all_weights["embeddings"] = {
				'user': tf.get_variable('user', shape=[n_users+1, n_embdims]),
				'item': tf.get_variable('item', shape=[n_items+1, n_embdims]),
				'type': tf.get_variable('type', shape=[n_types+1, n_embdims])
			}

		with tf.variable_scope(mlp_name, reuse=mlp_reuse):
			all_weights["weights"] = {
				'h1': tf.get_variable('h1', shape=[n_hidden_0, n_hidden_1],
					initializer=tf.contrib.layers.xavier_initializer()),
				'h2': tf.get_variable('h2', shape=[n_hidden_1, n_hidden_2],
					initializer=tf.contrib.layers.xavier_initializer()),
				'h3': tf.get_variable('h3', shape=[n_hidden_2, n_hidden_3],
					initializer=tf.contrib.layers.xavier_initializer())
			}
			all_weights['biases'] = {
				'b1': tf.get_variable('b1', shape=[n_hidden_1, ]),
				'b2': tf.get_variable('b2', shape=[n_hidden_2, ]),
				'b3': tf.get_variable('b3', shape=[n_hidden_3, ])
			}
		return all_weights

	def _MLP_f(self, X, keep_prob, embeddings, weights, biases):
		user_embedding = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings['user'], X[:,0]), axis=1, epsilon=1.0)
		item_embedding = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings['item'], X[:,1]), axis=1, epsilon=1.0)
		type_embedding = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings['type'], X[:,2]), axis=1, epsilon=1.0)

		mlp_input = tf.concat([user_embedding, item_embedding, type_embedding], 1)
		layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(mlp_input, weights['h1']), biases['b1'])), keep_prob)
		layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])), keep_prob)
		layer_3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])), keep_prob)

		if self.net_arch['n_layers'] == 1:
			output = layer_1
		elif self.net_arch['n_layers'] == 2:
			output = layer_2
		elif self.net_arch['n_layers'] == 3:
			output = layer_3

		return output

	def _MLP_g(self, X, keep_prob, embeddings, weights, biases):
		user_embedding = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings['user'], X[:,0]), axis=1, epsilon=1.0)
		type_embedding = tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings['type'], X[:,1]), axis=1, epsilon=1.0)

		mlp_input = tf.concat([user_embedding, type_embedding], 1)
		layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(mlp_input, weights['h1']), biases['b1'])), keep_prob)
		layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])), keep_prob)
		layer_3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])), keep_prob)

		if self.net_arch['n_layers'] == 1:
			output = layer_1
		elif self.net_arch['n_layers'] == 2:
			output = layer_2
		elif self.net_arch['n_layers'] == 3:
			output = layer_3

		return output

	def _create_loss_optimizer(self):
		pos_dist = tf.reduce_sum(tf.square(tf.subtract(self.anchor, self.pos)), 1)
		neg_dist = tf.reduce_sum(tf.square(tf.subtract(self.anchor, self.neg)), 1)

		pos_mask = tf.to_float(tf.equal(pos_dist, 0.0))
		neg_mask = tf.to_float(tf.equal(neg_dist, 0.0))

		pos_dist = tf.sqrt(pos_dist + pos_mask*1e-16)*(1.0 - pos_mask)
		neg_dist = tf.sqrt(neg_dist + neg_mask*1e-16)*(1.0 - neg_mask)
		basic = pos_dist - neg_dist

		self.push_loss = tf.reduce_mean(tf.maximum(tf.add(basic, self.alpha), 0.0), 0)
		self.pull_loss = tf.reduce_mean(pos_dist, 0)
		self.loss = self.push_loss
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def optimize_network(self, anchor, pos, neg, dropout_prob):
		self.sess.run(self.optimizer,
					feed_dict={self.anchor_x: anchor,
					self.pos_x: pos,
					self.neg_x: neg,
					self.dropout_x: dropout_prob})

	def get_action_embeddings(self, X):
		return self.sess.run(self.pos, feed_dict={self.pos_x: X, self.dropout_x:1.0})

	def get_anchor_embeddings(self, X):
		return self.sess.run(self.anchor, feed_dict={self.anchor_x: X, self.dropout_x:1.0})

	def build_minibatch(self, userkey, typekey, user_adjacency_dict, candidate_size):
		anchor = [userkey, typekey]
		anchor_embedding = self.get_anchor_embeddings([anchor])[0]

		# Obtain the action vectors for the positive items
		pos_items = list(user_adjacency_dict[typekey][userkey])
		pos_triples = [[userkey, item, typekey] for item in pos_items]
		pos_vectors = self.get_action_embeddings(pos_triples)

		# Sort the postive items in the descending order of their distances
		n_pos_items = len(pos_items)
		pos_distances = np.sqrt(np.sum(np.square(pos_vectors - anchor_embedding), axis=1))
		pos_dist_map = {pos_items[i] : pos_distances[i] for i in range(n_pos_items)}
		pos_items = heapq.nlargest(n_pos_items, pos_dist_map, key=pos_dist_map.get)

		# Obtain the action vectors for the negative items	
		neg_items = random.sample(range(1, self.net_arch['n_items']+1), candidate_size)
		for item in neg_items:
			if item in pos_items:
				neg_items.remove(item)
		neg_triples = [[userkey, item, typekey] for item in neg_items]
		neg_vectors = self.get_action_embeddings(neg_triples)

		# Sort the negative items in the descending order of their distances
		n_neg_items = len(neg_items)
		neg_distances = np.sqrt(np.sum(np.square(neg_vectors - anchor_embedding), axis=1))
		neg_dist_map = {neg_items[i] : neg_distances[i] for i in range(n_neg_items)}
		neg_items = heapq.nlargest(n_neg_items, neg_dist_map, key=neg_dist_map.get)	

		# Identify hard triplets using the sorted lists of the positive/negative items
		negidx_lb = 0
		batch_anchor, batch_pos, batch_neg = [], [], []
		for posidx in range(n_pos_items):
			while negidx_lb < n_neg_items and pos_dist_map[pos_items[posidx]] + self.alpha < neg_dist_map[neg_items[negidx_lb]]:
				negidx_lb += 1
			if negidx_lb == n_neg_items: break

			n_negsamples = min(self.n_negsamples, n_neg_items-negidx_lb)
			negidcs = random.sample(range(negidx_lb, n_neg_items), n_negsamples)
			for i in range(n_negsamples):
				batch_anchor.append(anchor)
				batch_pos.append([userkey, pos_items[posidx], typekey])
				batch_neg.append([userkey, neg_items[negidcs[i]], typekey])

		self.triplet_cnt += len(batch_anchor)
		return np.array(batch_anchor), np.array(batch_pos), np.array(batch_neg)

	def evaluate_triple(self, test_triple, user_adjacency_dict, item_adjacency_dict, ranking_set, K):
		userkey, itemkey, typekey = test_triple

		ranking_set = np.append(ranking_set, itemkey)
		ranking_triples = [[userkey, item, typekey] for item in ranking_set]
		ranking_vectors = self.get_action_embeddings(ranking_triples)

		anchor_embedding = self.get_anchor_embeddings([[userkey, typekey]])[0]
		square_distances = np.sum(np.square(ranking_vectors - anchor_embedding), axis=1)
		map_item_distance = {ranking_set[i] : square_distances[i] for i in range(len(ranking_set))}
		ranklist = heapq.nsmallest(K, map_item_distance, key=map_item_distance.get)	

		hr = getHitRatio(ranklist, itemkey)
		ndcg = getNDCG(ranklist, itemkey)

		return hr, ndcg
