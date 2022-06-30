For this project, you will be applying machine learning techniques that go beyond standard linear regression. 
You will have the opportunity to use a publicly available dataset to solve the problem of your choice. 
You are strongly discouraged from using well-known datasets, particularly ones that have been used as examples 
in previous courses or are similar to them (such as the iris, titanic, mnist, or movielens datasets, among others) - 
  this is your opportunity to branch out and explore some new data! The UCI Machine Learning Repository and Kaggle are 
good places to seek out a dataset. Kaggle also maintains a curated list of datasets that are cleaned and ready for 
machine learning analyses. Your dataset must be automatically downloaded in your code or included with your submission. 
You may not submit the same project for both the MovieLens and Choose Your Own project submissions.

The ability to clearly communicate the process and insights gained from an analysis is an important skill for data scientists. 
You will submit a report that documents your analysis and presents your findings, with supporting statistics and figures. 
The report must be written in English and uploaded as both a PDF document and an Rmd file. Although the exact format is up 
to you, the report should include the following at a minimum:
  
  an introduction/overview/executive summary section that describes the dataset and variables, and summarizes the goal of the 
project and key steps that were performed;
a methods/analysis section that explains the process and techniques used, including data cleaning, data exploration and 
visualization, insights gained, and your modeling approaches (you must use at least two different models or algorithms);
a results section that presents the modeling results and discusses the model performance; and
a conclusion section that gives a brief summary of the report, its potential impact, its limitations, and future work.
Your project submission will be graded both by your peers and by a staff member. 
The peer grading will give you an opportunity to check out the projects done by other learners. 
You are encouraged to give your peers thoughtful, specific feedback on their projects (i.e., more than just "good job" or 
                                                                                       "not enough detail").
# I decided to work on the breast cancer Wisconsin database
#breast cancer wisconsin
# let's install all we need#
install.packages("caret")
install.packages("ggplot2")
install.packages("tidyverse")
library(caret)
library(ggplot2)
library(tidyverse)

# let's start now#
getwd()
breast_cancer  <- read.csv(file = "wisconsin_breast_c_data.csv")
# let's check all variables
head(breast_cancer)
# let's convert in factors
breast_cancer$diagnosis <- as.factor(breast_cancer$diagnosis)
breast_cancer$diagnosis
# let's check structure
str(breast_cancer)
# we can see that the last column is full of NA
#let's check how many we have
sum(is.na(breast_cancer))
#is it only the one column we saw or are there more NAs?
nrow(breast_cancer)
# yes it seems as a full column only should be with NAs
# but let's check better by looking at missing elements
vis_miss(breast_cancer)
# yes, the last column is made by missing values#
# how many columns do we have? This way we will remove the last one
ncol(breast_cancer)
# let's remove the last one with NAs and check our new databse #
breast_cancer <- subset(breast_cancer[,-33])
breast_cancer

# now, how many columns do we have?
ncol(breast_cancer)
# 32, perfect!
# most interesting variable is diagnosis ($diagnosis) which is B for benign and M for malign
breast_cancer$diagnosis <- as.factor(breast_cancer$diagnosis)
breast_cancer$diagnosis
# Let's check how many malign and benign tumors we have
prop.table(table(breast_cancer$diagnosis))
# let's check a little bit more in the characteristics of the data and see if some predictors correlate more to a 
#specific diagnosis
# lets create breast_cancer_plot without first 2 columns which are Id and diagnosis (it is what we want to plot against diagnosis)
breast_cancer_plot <- breast_cancer[, -c(1,2)]
breast_cancer_plot
# Let's create breat_cancer_diag which is the diagnosis value we want to plot
breast_cancer_diag <- breast_cancer[,2]
breast_cancer_diag
# let's plot everything now and see if we have good predictors for diagnosis
scales <- list(x=list(relation="free"),y=list(relation="free"), cex=0.6)
featurePlot(x=breast_cancer_plot, y=breast_cancer_diag, plot="density", scales = scales,
            layout = c(3,10), auto.key = list(columns = 2), pch = "|", )
featurePlot
# wow we clearly see some predictors that have more impact on diagnosis than others
# these are for example: 
# concave points worst / perimeter worst / area worst /radius worst / concavity mean / concave points mean / area mean 
# compactness mean / radius mean / perimeter mean
#in machine learning it is good to use predictors which are not correlated with one another
# clearly some of the predictors are similar/should be correlated.
# So, let's check if we have any correlation between variables 
# to do so we will, once again, remove the id and diagnosis columns
breast_cancer_corr <- cor(breast_cancer %>% select(-id, -diagnosis))
breast_cancer_corr
#now let's plot correlations #
install.packages("corrplot")
library(corrplot)
corrplot::corrplot(breast_cancer_corr, order = "hclust", tl.cex = 1, addrect = 8)
# wow we have lots of correlated parameters, aka some variables are highly correlated with one another 
# (and hence are no good for training algorithms), while others are not correlated at all we will focus only on some of them

# let's blot them better and start a little machine learning using them
# let's check texture_mean 
ggplot(data= breast_cancer, aes(x = (diagnosis) , y = texture_mean)) + 
  geom_boxplot() 
# let's check concavity_mean
ggplot(data= breast_cancer, aes(x = (diagnosis) , y = concavity_mean)) + 
  geom_boxplot() 
# let's check are_mean
ggplot(data= breast_cancer, aes(x = (diagnosis) , y = area_mean)) + 
  geom_boxplot() 
# let's check compactness mean
ggplot(data= breast_cancer, aes(x = (diagnosis) , y = compactness_mean)) + 
  geom_boxplot() 
#let's check radius mean
ggplot(data= breast_cancer, aes(x = (diagnosis) , y = radius_mean)) + 
  geom_boxplot() 
# let's check perimeter mean
ggplot(data= breast_cancer, aes(x = (diagnosis) , y = perimeter_mean)) + 
  geom_boxplot() 
# let's check smoothness 
ggplot(data= breast_cancer, aes(x = (diagnosis) , y = smoothness_mean)) + 
  geom_boxplot() 

############################

#We are going to get a training and a testing set to use when building some models:
  
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = breast_cancer$diagnosis, times = 1, p = 0.1, list = FALSE)
breast_train <- breast_cancer[-test_index,]
breast_test <- breast_cancer[test_index,]

# Let's check the target variable and how they are represented in the test and training set.

prop.table(table(breast_train$diagnosis))*100
prop.table(table(breast_test$diagnosis))*100
# They're quite similar, it's a good thing
#Let's start applying machine learning models
# Let's start with logistic regression using all predictors and 5-fold cross val
# we know that using all predictors some will be highly correlated so from what we obtain using all predictors we will then 
#try to make it better
# Let's set the seed and the fit control, we will use 5-fold cross validation
fitControl <- trainControl(method="cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

# logistic regression using all predictors

logreg_all<-train(diagnosis~.,data=breast_train[,-1],method="glm",family=binomial(),
            trControl=fitControl)
# let's see variable importance
varImp(logreg_all)

#Testing the logistic regression with all predictors model.

logreg_all_pred<-predict(logreg_all,breast_test[,-c(1,2)])
cm_logreg_all<-confusionMatrix(logreg_all_pred,breast_test$diagnosis)
cm_logreg_all
##### great, good accuracy!!!! 0.9138!!!
# but we can do better we know that some variables are more predictive of diagnosis than others and we know that some variables
# are higly correlated: not a good thing, let's try to implement a bit
head(breast_cancer)
# Let's try using only 2 variables with high importance but that are not correlated
logreg_area_compactness<-train(diagnosis~area_mean+compactness_mean, data=breast_train[,-1],method="glm", family=binomial(),
                  trControl=fitControl)
varImp(logreg_area_compactness)

#Testing the logistic regression with only two good predictors

logreg_area_compactness_pred<-predict(logreg_area_compactness,breast_test)
cm_logreg_area_compactness<-confusionMatrix(logreg_area_compactness_pred,breast_test$diagnosis)
cm_logreg_area_compactness
#Cool it is much better now, around 0.95 of accuracy, using less predictors is actually better
# Now let's try something totally different LDA
# LDA is bla bla bla
# LDA
model_breast_lda <- train(diagnosis~.,
                   breast_train,
                   method="lda2",
                   #tuneLength = 10,
                   metric="ROC",
                   preProc = c("center", "scale"),
                   trControl=fitControl)
pred_breast_lda <- predict(model_breast_lda, breast_test)
cm_lda <- confusionMatrix(pred_breast_lda, breast_test$diagnosis, positive = "M")
cm_lda
#good we get accuracy of 0.931 but not as good as our previous model
#Let's try random forest
# random forest 
installpackages(ranger)
library("ranger")
model_breast_ranger  <- train(diagnosis ~ ., data = breast_train[,-1], method = "ranger")
# Let's check the model
model_breast_ranger
# testing random forest model
pred_breast_ranger <- predict(model_breast_ranger, breast_test[,-c(1,2)])
cm_ranger <- confusionMatrix(pred_breast_ranger, breast_test$diagnosis)
cm_ranger
# Wow! this is the best so far we got an accuracy of 0.9828
# and really high sensitivity and specificity!!! perfect!!!! this is our final model!!!

#Let's try one last method though: KNN -> K nearest neighbours
# While transforming the data I excluded the target variable, diagnosis but for KNN 
#model training I need to store the target variable splitting between training and testing set.

knn5_breast_fit <- knn3(diagnosis ~ ., data = breast_train, k=5)
pred_breast_knn5 <- predict (knn5_breast_fit, breast_test, type= "class")
cm_breast_knn5 <- confusionMatrix(pred_breast_knn5, breast_test$diagnosis)
cm_breast_knn5
# pretty bad this way only 0.7241 accuracy
# Let's try modifying the number of K, let's do 3 
knn3_breast_fit <- knn3(diagnosis ~ ., data = breast_train, k=3)
pred_breast_knn3 <- predict (knn3_breast_fit, breast_test, type= "class")
cm_breast_knn3 <- confusionMatrix(pred_breast_knn3, breast_test$diagnosis)
cm_breast_knn3
# A little better but still bad, let's stick with previous models
# Best model is the one that used logistic regression using area and compactness as predictors
