

## ---- libraries ----
## Load the required libraries.
library(caret); library(class); library(nnet)
library(e1071); library(randomForest); library(gbm)

## ---- read_data ----
## Import the data.
full.data  <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing    <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

## ---- process_data ----
## Remove all columns where the entire column in the testing data.frame is NA.
empty      <- colSums(is.na(testing))
remove     <- which(empty == 20)
clean.data <- full.data[, - remove]
clean.test <- testing[, - remove]

## ---- create_validation_set ----
## Create a training and testing dataset.
set.seed(55)
index      <- createDataPartition(y = full.data$classe, p = 0.7, list = FALSE)
training   <- clean.data[index, ]
validation <- clean.data[- index, ]

## ---- k-means_classification ----
## Scale the data because the k-nearest neighbor is done by Euclidian distance.
knn.training   <- scale(training[, 8:59])
knn.class      <- training[, 60]
knn.validation <- scale(validation[, 8:59])

## K-nearest neighbor classification for 
k1.prediction <- knn(knn.training, knn.validation, knn.class, k = 1, prob = TRUE)
k2.prediction <- knn(knn.training, knn.validation, knn.class, k = 2, prob = TRUE)
k3.prediction <- knn(knn.training, knn.validation, knn.class, k = 3, prob = TRUE)
k4.prediction <- knn(knn.training, knn.validation, knn.class, k = 4, prob = TRUE)
k5.prediction <- knn(knn.training, knn.validation, knn.class, k = 5, prob = TRUE)

## Calculate the error.
sum(k1.prediction == validation[, 60]) / length(k1.prediction)
sum(k2.prediction == validation[, 60]) / length(k2.prediction)
sum(k3.prediction == validation[, 60]) / length(k3.prediction)
sum(k4.prediction == validation[, 60]) / length(k4.prediction)
sum(k5.prediction == validation[, 60]) / length(k5.prediction)

## ---- logistic_regression -----
##  
mlogit.model      <- multinom(classe ~., family = 'multinomial', data = training[, 8:60])
mlogit.prediction <- predict(mlogit.model, newdata = validation[, 8:60])
sum(mlogit.prediction == validation[, 60]) / length(mlogit.prediction)

## ---- linear_discriminant_analysis ----
##
lda.model      <- lda(classe ~., data = training[, 8:60])
lda.prediction <- predict(lda.model, newdata = validation[, 8:60])
sum(lda.prediction$class == validation[, 60]) / length(lda.prediction$class)

## ---- quadratic_discriminant_analysis ----
##
qda.model      <- qda(classe ~., data = training[, 8:60])
qda.prediction <- predict(qda.model, newdata = validation[, 8:60])
sum(qda.prediction$class == validation[, 60]) / length(qda.prediction$class)

## ---- support_vector_machine ----

svm.linear <- svm(classe ~., kernel = "linear", data = training[, 8:60], scale = TRUE)
svm.poly   <- svm(classe ~., kernel = "polynomial", data = training[, 8:60], scale = TRUE)
svm.radial <- svm(classe ~., kernel = "radial", data = training[, 8:60], scale = TRUE)

svm.linear.prediction <- predict(svm.linear, newdata = validation[, 8:60])
sum(svm.linear.prediction == validation[, 60]) / length(svm.linear.prediction)

svm.poly.prediction <- predict(svm.poly, newdata = validation[, 8:60])
sum(svm.poly.prediction == validation[, 60]) / length(svm.poly.prediction)

svm.radial.prediction <- predict(svm.radial, newdata = validation[, 8:60])
sum(svm.radial.prediction == validation[, 60]) / length(svm.radial.prediction)

## ---- random_trees ----
## Bagging classification tree
set.seed(40)
bag.model      <- randomForest(classe ~ ., data = training[, 8:60], mtry = 52)
bag.prediction <- predict(bag.model, newdata = validation[, 8:60])
sum(bag.prediction == validation[, 60]) / length(bag.prediction)

## Random forest model
set.seed(41)
random.forest            <- randomForest(classe ~ ., data = training[, 8:60], mtry = 7)
random.forest.prediction <- predict(random.forest, newdata = validation[, 8:60])
sum(random.forest.prediction == validation[, 60]) / length(random.forest.prediction)



