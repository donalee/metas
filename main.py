import tensorflow as tf
import os, sys
from utils import read_file
from train import train

flags = tf.app.flags

flags.DEFINE_string("data_dir", "./data/", "The path to the input dataset")
flags.DEFINE_string("dataset", "tmall", "The name of the input dataset")
flags.DEFINE_string("n_hiddens", "250 250 250", "The sizes of hidden layers")

flags.DEFINE_integer("n_embdims", 250, "The size of the entity embedding vectors")
flags.DEFINE_integer("n_layers", 1, "The number of hidden layers")
flags.DEFINE_integer("n_negsamples", 3, "The number of negative items per an observed triple.")
flags.DEFINE_integer("n_updates", 5000000, "The nubmer of updates")
flags.DEFINE_integer("batch_size", 200, "The number of triplets in a mini-batch")
flags.DEFINE_integer("display_step", 5000, "The display step")
flags.DEFINE_integer("eval_at", 10, "The number of top-k items to be retrieved")

flags.DEFINE_float("alpha", 2.0, "The margin size for the triplet loss")
flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate.")
flags.DEFINE_float("dropout_prob", 0.5, "The dropout keep probability")

flags.DEFINE_boolean("gpu", False, "Enable to use GPU for training, instead of CPU")
flags.DEFINE_integer("gpu_devidx", 0, "The device index of the target GPU (in case that multiple GPUs are available)")

def main():
	FLAGS = flags.FLAGS

	n_entities, user_type_pairs, user_adj_dict, item_adj_dict, test_data, valid_data, rankingset_data = read_file(FLAGS.data_dir + FLAGS.dataset)
	n_users, n_items, n_types = n_entities

	if not FLAGS.gpu:
		os.environ["CUDA_VISIBLE_DEVICES"] = ""
	else:
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_devidx)

	network_architecture = dict(
				n_users = n_users,		
				n_items = n_items, 	
   				n_types = n_types,	
	   			n_embdims = FLAGS.n_embdims,
				n_layers = FLAGS.n_layers,
				n_hiddens = FLAGS.n_hiddens)

	train(network_architecture, user_type_pairs, user_adj_dict, item_adj_dict,
				test_data, valid_data, rankingset_data,
				alpha=FLAGS.alpha, learning_rate=FLAGS.learning_rate, 
				n_updates=FLAGS.n_updates, batch_size=FLAGS.batch_size, 
				n_negsamples=FLAGS.n_negsamples, dropout_prob=FLAGS.dropout_prob, 
				display_step=FLAGS.display_step, K=FLAGS.eval_at)

if __name__ == '__main__':
	main()
