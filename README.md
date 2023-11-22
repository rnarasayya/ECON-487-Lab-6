# ECON-487-Lab-6
1.	Estimate a random forest model and compare the MSE with the same LASSO model when predicting sales.  Try to make a complicated model.  Remember that you have to install the randomForest package.
a.	Here is some code: 
Install and use the randomForest package. 
mydata$price <- log(mydata$price) 
oj.rf <- randomForest(logmove ~ ., data = mydata, ntree = 	100, keep.forest = TRUE) 
mydata$pred_logmove_rf = predict(oj.rf) 
mydata$resid2 <- 	(mydata$logmove - mydata$pred_logmove_rf)^2 
b.	Try to plot observed versus predicted using ggplot.      
c.	Compare to your complicated LASSO model from the previous problem set for the MSE. Remember to hold out data so your random forest MSE is fair!
2.	We’re going to do some basic exploration with xgboost. 
a.	Install the package xgboost and library it.
b.	Divide the data into a training set (80% of the data) and a hold-out set (20% of the data). 
c.	We’re going to train a model to predict logmove. To do this, we’re going to create a training and testing matrix that we can give to the package to do cross validation on. 
i.	Use the xgb.DMatrix function to create a train and test matrix. This function takes arguments “data” (must be a matrix, so consider using the model.matrix command) and “label” (the outcome, logmove in our case).
ii.	Use the xgb.cv function to do 5-fold cross-validation on our training data. We’ll just use the defaults for most of the hyperparameters. A few useful arguments:
1.	nfold: number of folds for cross-validation
2.	nrounds: number of training rounds (generally, we want this to be a very large number since we don’t want to be artificially stopped short of achieving a minimum)
3.	early_stopping_rounds: if this argument is set, XGBoost will stop training if the testing error does not improve in whatever number the user puts here. This should be our stopping criterion (as opposed to hitting nrounds)
4.	print_every_n: if you set this to, say, 100, XGBoost will report its progress every 100 iterations, instead of each iteration.
5.	Important note: we’re not actually cross-validating or setting any of the hyperparameters that make XGBoost a powerful algorithm. If you’re curious about what other parameters you can set, inspect the documentation for this function or for the function xgboost.
iii.	Report the training RMSE (root mean squared error) and testing RMSE from the best model. How does this compare to previous models that we’ve used (remember that you should square this to get MSE)?
iv.	Use the xgboost function to train a model on the full training data using our one cross-validated hyperparameter (the number of training iterations). To do this, find the best iteration of the cross validated model and set that as nrounds for the xgboost function.
v.	Use the predict command (the same way that we do in regression) and your testing xgb.DMatrix to assess the fit of the model on the held out data. How does the MSE compare to the MSE from cross-validation? How does it compare to prior models?

