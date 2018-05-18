from __future__ import print_function

import pandas as pd          
import numpy as np
import tensorflow as tf
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2`' 

learning_rate = 0.001
num_steps = 100
mini_batch_size = 100
display_step = 1

n_hidden_1  = 8
n_hidden_2  = 8
n_hidden_3 = 8
num_input  = 13
num_classes = 2

output_file = open("output.txt","w+")

# Might consider making changes to evaluation or minimization
# Standard Deviation of 0.02 works 13 13 8 or 13 13 13

def get_accuracy(predictions, labels):
	preds_correct_boolean = np.equal(predictions,labels)
	print(preds_correct_boolean)
	correct_predictions = np.sum(preds_correct_boolean)
	print(correct_predictions)
	acc = 100.0 * (correct_predictions / preds_correct_boolean.shape[0])
	return acc

def Logistic(X,T):
	logistic = LogisticRegression()
	a = logistic.fit(X,T)
	y_pred = logistic.predict(X)
	ac = accuracy_score(y_pred,T)
	return ac

def nn(X,T2,X_test,Y_test):
	
	# tf Graph input
	x = tf.placeholder("float", [None, num_input])
	y = tf.placeholder("float", [None, num_classes])

	# Store layers weight & bias
	weights = {
		'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1],dtype=np.float32,stddev = 0.002)),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], dtype=np.float32,stddev = 0.002)),	
		'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes], dtype=np.float32,stddev = 0.002))
	}
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1], dtype=np.float32,stddev = 0.002)),
		'b2': tf.Variable(tf.random_normal([n_hidden_2], dtype=np.float32,stddev = 0.002)),
		'out': tf.Variable(tf.random_normal([num_classes], dtype=np.float32,stddev = 0.002))
	}

	# Create model

	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	y_layer_1 = tf.nn.relu(layer_1) #tf.nn.sigmoid
	layer_2 = tf.add(tf.matmul(y_layer_1, weights['h2']), biases['b2'])
	y_layer_2 = tf.nn.relu(layer_2)
	out_layer = tf.matmul(y_layer_2, weights['out']) + biases['out']

	# Construct model
	logits = out_layer
	prediction = tf.nn.softmax(logits)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
	#loss_op = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	#grads_and_vars = optimizer.compute_gradients(loss_op,tf.trainable_variables())
	#train_op = optimizer.apply_gradients(grads_and_vars)
	train_op = optimizer.minimize(loss_op)

	# Evaluate model
	qq = tf.argmax(prediction, 1)
	correct_pred = tf.equal(qq, tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	t_acc = 0

	# Start training
	with tf.Session() as sess:

		# Run the initializer
		sess.run(init)
		fin_acc = 0
		for step in range(1, num_steps+1):
			offset = (step * mini_batch_size) % (X.shape[0] - mini_batch_size)
			batch_x = X[offset:(offset + mini_batch_size), :]
			batch_y = T2[offset:(offset + mini_batch_size)]


			# Run optimization op (backprop)
			sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
			#gr_print = sess.run([grad for grad, _ in grads_and_vars], feed_dict={x:batch_x, y : batch_y})
			#print(gr_print)

			loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_x,y: batch_y})
			
			print("Step " + str(step) + ", Minibatch Loss= " + "{:.9f}".format(loss) + ", Training Accuracy= " + "{:.9f}".format(acc))


		fin_acc = sess.run(accuracy, feed_dict={x: X,y: T2})
		writer = tf.summary.FileWriter('logs', sess.graph)

		print("Training Accuracy is")
		print(fin_acc)
		print("Testing Accuracy is")
		t_acc = sess.run(accuracy,feed_dict = {x:X_test,y:Y_test})
		print(t_acc)

	return t_acc

if __name__ == "__main__":

	data = pd.read_csv("Bank_EXIT_Survey.csv")
	feature_names = list(data)

	data = data[feature_names[3:]]
	target = data["Status"]
	data.drop("Status",axis=1,inplace=True)

	data2 = pd.get_dummies(data["City"])
	result = pd.concat([data, data2], axis=1)
	result.drop("City",axis=1,inplace=True)

	data2 = pd.get_dummies(result["Gender"])
	result = pd.concat([result,data2],axis=1)
	result.drop("Gender",axis=1,inplace=True)

	X = result.values
	T = target.values

	res = tf.one_hot(indices = T,depth = 2)

	with tf.Session() as sess:
		T2 = res.eval()

	print("Accuracy for Logistic Regression is :")
	print(Logistic(X,T))

	k_fold_list = [5,10]
	mean_accuracies = []

	for k_fold in k_fold_list:

		output_file.write("For " + str(k_fold) + " Validation:\n")

		cv = KFold(n_splits=k_fold)
		accuracies = []

		for train_index, test_index in cv.split(X):
			training_X = X[train_index]
			training_y = T2[train_index]
			testing_X = X[test_index]
			testing_y = T2[test_index]
			accuracies.append(nn(training_X,training_y,testing_X,testing_y))

		output_file.write("The accuracies are:\n")
		output_file.write(str(accuracies) + "\n")

		output_file.write("The mean accuracy is :")
		mean_accuracies.append(np.mean(accuracies))
		output_file.write(str(np.mean(accuracies)))
		output_file.write("\n")

	print(mean_accuracies)
	output_file.write("Mean accuracies are ")
	output_file.write(str(mean_accuracies))