#Clear Workspace
rm(list = ls())

# Necessary libraries
library(caret)
library(randomForest)
library(party)
library(gbm)
library(class)

# Set seed for reproducibility and also set working directory
set.seed(1)
setwd("C:/Users/Dimas/Documents/R/KaggleTitanic/")

# Loading data
load("TestData.rda")
load("RawData.rda")

# Partitioning RawData into training and testing sets
inTrain = createDataPartition(RawData$Survived, p = 0.8)[[1]]
training <- RawData[inTrain,]
testing <- RawData[-inTrain,]


###########
###########
###########
###########
###########



# Training the Random Forest model
Rf <- randomForest(Survived~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2, 
                        prox = TRUE, 
                        data = RawData, 
                        ntree = 2000, 
                        mtry = 3,
                        importance = TRUE)
# Saving the model
save(Rf, file = "Rf.rda")

# Loading the model when we have one saved already
load("Rf.rda")

# We do not need to use cross validation with RF, good approximation is:
predictionRF <- predict(Rf, newdata = TestData)
confusionMatrix(predict(Rf), training$Survived)



###########
###########
###########
###########
###########


fitControl <- trainControl(method = "repeatedcv",        ## do repeated Cross Validation
                           classProbs = TRUE,
                           number = 10,                  ## 10-fold
                           repeats = 20)                 ## ten times



#########
# Logistic regression with 4-fold cross-validation

Lfit <- train(Survived~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2, 
              data=RawData,
              method="glm",
              trControl=fitControl,
              family=binomial)
# Saving the model
save(Lfit, file = "Lfit.rda")

# Loading the model when we have one saved already
load("Lfit.rda")

# Now test the model on the test set, accuracy = 0.79
predictionLM <- predict(Lfit, newdata = testing, type = "prob")
confusionMatrix(testing$Survived, predictionLM)


####################################################

# Training the Naive Bayes model with 4-fold cross-validation repeated 10 times           
NBfit <- train(Survived~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilyID2, 
               data=training,
               method="nb",
               trControl=fitControl,
               family=binomial)

# As we can see, the 4-fold cross-validated accuracy is about 0.573
NBfit$results
mean(NBfit$resample[,1])
# Saving the model
save(NBfit, file = "NBfit.rda")

# Loading the model when we have one saved already
load("NBfit.rda")

# Now test NB model on the test set, accuracy = 0.595
predictionNB <- predict(NBfit, newdata = testing, type = "prob")
confusionMatrix(testing$Survived, predictionNB)


#####################################################

Ctree <- cforest((Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
               data = RawData, 
               controls=cforest_unbiased(ntree=2000, mtry=3))
# Saving the model
save(Ctree, file = "Ctree.rda")

# Loading the model when we have one saved already
load("Ctree.rda")
predictionCtree <- predict(Ctree, newdata = TestData)
confusionMatrix(testing$Survived, predictionCtree)

# Probabilities of survival from Ctree
presp<- treeresponse(Ctree, newdata = testing)


####################
####################
####################
####################
####################



gbmGrid <-  expand.grid(interaction.depth = c(2),
                        n.trees = c(3:18)*50,
                        shrinkage = 0.1)

GBMfit1 <- train((Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
                 data = RawData,
                 method = "gbm",
                 trControl = fitControl,
                 tuneGrid = gbmGrid,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
save(GBMfit1, file = "GBMfit1.rda")
# Cross-validated accuracy is 0.582
GBMfit1$results[9,]

# Loading the model when we have one saved already
load("GBMfit1.rda")

# Now test GBM model on the test set, accuracy = 0.63
predictionGBM <- (predict(GBMfit1, testing)) 
confusionMatrix(testing$Survived, predictionGBM)


####################
####################
####################
####################
####################



# Making a final prediction, probability of 0 (=not survived)

p1RF<- predict(Rf, newdata = RawData)
temp<- treeresponse(Ctree, newdata = RawData)
p2CT <- predict(Ctree,newdata = RawData)
p3LM<- predict(Lfit, newdata = RawData)
p4GBM<- predict(GBMfit1, newdata = RawData)
p5NB<-predict(NBfit, newdata = RawData)



TrainPred <- data.frame(cbind(p1RF, p2CT,p3LM,p4GBM,p5NB, RawData$Survived))
for (i in 1:6){TrainPred[,i] <- factor(TrainPred[,i])}

finalModel <- train(V6~., 
                    data = TrainPred, 
                    method="nb",
                    trControl=fitControl,
                    family=binomial)








p1RF<- predict(Rf, newdata = TestData)
temp<- treeresponse(Ctree, newdata = TestData)
p2CT <- predict(Ctree,newdata= TestData)
p3LM<- predict(Lfit, newdata = TestData)
p4GBM<- predict(GBMfit1, newdata = TestData)
p5NB<-predict(NBfit, newdata = TestData)



TestPred <- data.frame(cbind(p1RF, p2CT,p3LM,p4GBM,p5NB, TestData$Survived))
for (i in 1:6){TestPred[,i] <- factor(TestPred[,i])}

TestPred$Survived <- predict(finalModel, TestPred)
levels(TestPred$Survived) <- c(0,1)

Data <- read.csv(file = "test.csv")
submission <- data.frame(Data[,1], TestPred$Survived)
colnames(submission) = c("PassengerId", "Survived")
write.csv(submission, "5Models.csv", row.names=FALSE)

