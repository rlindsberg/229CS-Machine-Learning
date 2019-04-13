Week2 Classification and Representation

Email: spam/not spam
Online Transactions> fraudulent or not
Tumor: malignant/benign

Logistic Regression (classification algoritsm)

Sigmoid/logistic function
htheta(x)=g(theta.T*x)
g(z)= 1/(1+e^(-z))

3.6
Optimization algorithsm
Took Andrew a decade to understand what they do...
More advanced algorithsms:
Conjugate gradient
BFGS
L-BFGS

Pros: no need ot manually pick alpha, often faster than gradient descent
Cons: more complex


Regulation, prevent overfitting
Overfit - high bias
Overfit - high variance
If we have too many features, the learned hypothesis may fit the training set very well, J= 0, but fail to generalize to new examples. (tried too hard)

Options:
Reduce number of features
Regularization
