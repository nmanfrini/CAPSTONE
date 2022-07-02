 

# For my capstone final project I decided to work on a breast cancer dataset obtained from Kaggle                                                                                                                                                                             "not enough detail").
# Specifically I decided to work on the "Breast Cancer Wisconsin" dataset.
# It is a dataset with various features that are computed from a digitized image of a fine needle aspirate (FNA) 
# of a breast mass.
# Specifically the fetures describe characteristics of the cell nuclei present in the images of breast cancer cells analyzed.
# Data were published along with the paper of K. P. Bennett and O. L. Mangasarian 
# "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", on the "Optimization Methods and Software"
# journal in 1992 (Volume 1, pages 23-34).
# For each cell nucleus they give: 
# 1) ID number
# 2) Diagnosis for the specific cell/nucleus sample.
# For each analyzed nucleus they also give the following values/infos:

#a) radius (mean of distances from center to points on the perimeter)
#b) texture (standard deviation of gray-scale values)
#c) perimeter
#d) area
#e) smoothness (local variation in radius lengths)
#f) compactness (perimeter^2 / area - 1.0)
#g) concavity (severity of concave portions of the contour)
#h) concave points (number of concave portions of the contour)
#i) symmetry
#j) fractal dimension ("coastline approximation" - 1)
# For all of these features authors computed the mean, standard error and "worst" (mean of the three
#largest values), giving rise to a total of 30 features

# CLEARLY the idea would be to predict diagnosis of the sample (malign, M or benign, B) from one, some or all of
#the infos given in points a-j
#We're in gear, LET'S START
# First thing, let's install all we need#
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")

library(caret)
library(ggplot2)
library(tidyverse)

# let's start now#
#Let's check our working directory to be sure where our dataset file is stored
getwd()
#Let's upload the dataset
breast_cancer  <- read.csv(file = "wisconsin_breast_c_data.csv")
# let's check all variables
head(breast_cancer)
#we will be interested in the diagnosis variable so will convert it in to factor to ease all of the analysis below
# let's convert in factors
breast_cancer$diagnosis <- as.factor(breast_cancer$diagnosis)
# Let's check it
breast_cancer$diagnosis
# let's check the structure of the database to see if it is tidy or if we have to tidy it up a little
str(breast_cancer)
# we can see that the last column is full of NA
#let's check how many NAs we have
sum(is.na(breast_cancer))
#is seems to be only the one column. Let's make sure the length of the column in the dataset is the same as the number of NAs
nrow(breast_cancer)
# yes, it is the same.  It seems as if only one full column should be filled with NAs
# but let's check better by looking at missing elements
#to do so we need the naniar package
if(!require(naniar)) install.packages("naniar", repos = "http://cran.us.r-project.org")
library(naniar)
vis_miss(breast_cancer)
# yes, the last column is made by missing values#
# how many columns do we have? This way we will remove the last one with all the NAs
ncol(breast_cancer)
# let's remove the last one with NAs and check our new database #
breast_cancer <- subset(breast_cancer[,-33])
breast_cancer
# now, how many columns do we have? (should be 32)
ncol(breast_cancer)
# 32, perfect! Let's check for residual missing values
vis_miss(breast_cancer)
# perfect, we're clean and good to go!!
# The most interesting variable is diagnosis ($diagnosis) which is B for benign and M for malign
#let's have a look
breast_cancer$diagnosis <- as.factor(breast_cancer$diagnosis)
breast_cancer$diagnosis
# Let's check how many malign and benign tumors we have overall (in percentage)
prop.table(table(breast_cancer$diagnosis))
# let's check a little bit more in the characteristics of the data and see if some predictors correlate more to a 
#specific diagnosis
# lets create breast_cancer_plot without first 2 columns which are Id and diagnosis (it is what we want to plot against 
# diagnosis)
breast_cancer_plot <- breast_cancer[, -c(1,2)]
breast_cancer_plot
# Let's create breast_cancer_diag which is the diagnosis value we want to plot
breast_cancer_diag <- breast_cancer[,2]
breast_cancer_diag
# let's plot everything now and see if we have good predictors for diagnosis
scales <- list(x=list(relation="free"),y=list(relation="free"), cex=0.6)
featurePlot(x=breast_cancer_plot, y=breast_cancer_diag, plot="density", scales = scales,
            layout = c(2,5), auto.key = list(columns = 2), pch = "|", )
featurePlot
# wow!! We clearly see some predictors that have more impact on diagnosis than others
# these are for example: 
# concave points worst / perimeter worst / area worst /radius worst / concavity mean / concave points mean / area mean 
# compactness mean / radius mean / perimeter mean
#In machine learning it is good to use predictors which are not correlated with one another
# clearly some of the predictors are similar/should be correlated.
# So, let's check if we have any correlation between variables 
# to do so we will, once again, remove the id and diagnosis columns
breast_cancer_corr <- cor(breast_cancer %>% select(-id, -diagnosis))
breast_cancer_corr
#now let's plot correlations, to do so we will need the corrplot package#
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
library(corrplot)
corrplot::corrplot(breast_cancer_corr, order = "hclust", tl.cex = 1, addrect = 8)
# wow we have lots of correlated parameters, aka some variables are highly correlated with one another 
#while others are not correlated at all we will focus only on some of them.
# using highly correlated variables in general is not good for machine learning
# let's blot some of the variables better in order to see their impact on diagnosis 
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
# all of these variables seem to positively correlate with a malignant diagnosis
#Let's start a little machine learning
#First thing we need a training and a test set:
  
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = breast_cancer$diagnosis, times = 1, p = 0.1, list = FALSE)
breast_train <- breast_cancer[-test_index,]
breast_test <- breast_cancer[test_index,]

# Let's check the target variable (diagnosis) and see how it is represented in the test and training set.

prop.table(table(breast_train$diagnosis))*100
prop.table(table(breast_test$diagnosis))*100
# The selected variable is represented similarly in train and test sets. It's a good thing
#Let's start applying machine learning models
# Let's start with logistic regression using all predictors and 5-fold cross validation
# We know that using all predictors some of them will be highly correlated to one another and this is not a good thing
# for machine learning, so from what we obtain using all predictors we will then try to make it better
# Let's set the fit control with a  5-fold cross validation.
fitControl <- trainControl(method="cv",
                           number = 5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

# logistic regression using all predictors

logreg_all<-train(diagnosis~.,data=breast_train[,-1],method="glm",family=binomial(),
            trControl=fitControl)
# let's see variable importance
varImp(logreg_all)

#Perfect, now let's test the "logistic regression with all predictors" model on the test set.

logreg_all_pred<-predict(logreg_all,breast_test[,-c(1,2)])
cm_logreg_all<-confusionMatrix(logreg_all_pred,breast_test$diagnosis)
cm_logreg_all
#great, good accuracy!!!! 0.9138!!!
# but we can do better we know that some variables are more predictive of diagnosis than others and 
#we know that some variables are higly correlated: not a good thing to train models, let's try to implement a bit
# Let's try using only 2 variables with high importance but that are not correlated
logreg_area_compactness<-train(diagnosis~area_mean+compactness_mean, data=breast_train[,-1],method="glm", family=binomial(),
                  trControl=fitControl)
varImp(logreg_area_compactness)

#Now let's test the "logistic regression with only two good predictors" model on the test set.

logreg_area_compactness_pred<-predict(logreg_area_compactness,breast_test)
cm_logreg_area_compactness<-confusionMatrix(logreg_area_compactness_pred,breast_test$diagnosis)
cm_logreg_area_compactness
#Cool!! it is much better now, around 0.95 of accuracy, using less predictors is actually better
# Since using less predictors is actually better, let's try using a different approach: LDA
# LDA (Linear Discriminant Analysis) is a relatively simple solution to the problem of having too many parameters 
# as it to assumes that the correlation structure is the same for all classes.
# This approach reduces the number of parameters to be estimated.
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
#good we get accuracy of 0.931, better than regression with alla variables but not as good as our previous model 
#(using only 2 good predictors)
#Let's try random forest
# random forest 
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
library("ranger")
model_breast_ranger  <- train(diagnosis ~ ., data = breast_train[,-1], method = "ranger")
# Let's check the model a bit
model_breast_ranger
# Let's test it
pred_breast_ranger <- predict(model_breast_ranger, breast_test[,-c(1,2)])
cm_breast_ranger <- confusionMatrix(pred_breast_ranger, breast_test$diagnosis)
cm_breast_ranger
# Wow! AWESOME!!!!! This is the best so far, we got an accuracy of 0.9828
# and really high sensitivity and specificity!!! perfect!!!! this is our final model!!!
# We probably can't do better than this, but
# For curiosity, let's try one last method: KNN -> K nearest neighbours using k= 5

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
# Let's plot them all together
model_results <- bind_rows(data_frame(method= "cm_logreg_all", accuracy = cm_logreg_all$overall["Accuracy"]),
                           data_frame(method= "cm_logreg_area_compactness", accuracy = cm_logreg_area_compactness$overall["Accuracy"]),
                           data_frame(method= "cm_lda", accuracy = cm_lda$overall["Accuracy"]),
                           data_frame(method= "cm_breast_ranger", accuracy = cm_breast_ranger$overall["Accuracy"]),
                           data_frame(method= "cm_breast_knn5", accuracy = cm_breast_knn5$overall["Accuracy"]),
                           data_frame(method= "cm_breast_knn3", accuracy = cm_breast_knn3$overall["Accuracy"]))
model_results
# The best models are: RANDOM FOREST and logistic regression using area and compactness as predictors.
# we did great!!!!!
# Last thing I checked in detail specificity and sensitivity of the two best models, as these features are of 
#fundamental importance to evaluate how good a model is.
cm_logreg_area_compactness
cm_lda
cm_breast_ranger
# they are really high in all cases, but once again RANDOM FOREST IS BETTER
# BEST MODEL: RANDOM FOREST!!
#Looking at the two values for each model once again RANDOM FOREST wins, it is the best overall
#This is great. I generated a really good model to predict if a breast cancer cell is malignant or benign 
#based on info regarding the nucleus.
