
Customer-Retention-Using-Neural-Network
====================================================
Creation of a MultiLayer Perceptron using Back Propagation Algorithm. It was trained to efficiently classify the data into two sets: exit and stay. This was able to predict whether a customer might stay with the bank or leave it in future.

Data Description:
----------------

- Dataset Size – 10000
- Number of features – 13
- Target – Classification – Yes (1 – Continued to work) or No (0 – Did not continue to work)

Preprocessing
------------------
In the 13 features, 3 features cannot be used for prediction due to the fact that the ids and names do not represent any inherent quality of the customer and also all the only one data point per id is present. Therefore, we cannot learn anything from the ids. Therefore, we can safely remove the features UID, customer name and RowNumber from the dataset that we use for prediction. Any pattern that we find using these ids would be spurious.

All the other features can be used for our classification task as they provide us information about the general targets with respect to the feature values. The variables like City, Gender are qualitative variables and do not have any ordering over them. Therefore, these variables have to be converted to dummy variables and have to be incorporated.

All other variables are quantitative variables and can be directly used for prediction.

Results
---------
- The average accuracies achieved were the same in both the cases of 5 fold cross validation and 10 fold cross validation with a mean accuracy of 79.63% ~ 80%.
- The accuracies of the 5 fold cross validation are : [0.792, 0.785, 0.7985, 0.801, 0.805]
- The accuracies of the 10 fold cross validation are : [0.796, 0.788, 0.784, 0.786, 0.8,0.797, 0.806, 0.796, 0.789, 0.821]

We had experimented with various activation functions (relu, identity, sigmoid,tanh) and we have got the same results for all the above cases. But experimentation on the data set has shown that the gradients completely vanish after a point and then, we are left with same weight vectors and therefore same errors. 

Since the gradients are small, Relu activation function is used which is said to reduce the vanishing gradient problem in Neural Networks and also makes training faster. The output layer uses softmax activation function because it is the best activation function for multi-class classification. 

The choosing of number of layers and number of nodes in hidden layers is generally selected using experimentation. But in this case, there is no difference in the loss even if we changed the architecture. Therefore we go with a minimal two hidden layers and around 8 neurons per hidden layer.
