# Clear Workspace
rm(list = ls())

# Necessary libraries
library(caret)
library(rpart)

# Set seed for reproducibility and also set working directory
set.seed(1)
setwd("C:/Users/Dimas/Documents/R/KaggleTitanic/")

# Reading data
RawData <- read.csv(file = "train.csv")
TestData <- read.csv(file = "test.csv")


# As we are going to apply same data processing to training and test sets, 
# it makes sense to merge them, process and then separate again. This way 
# we'll get rid of repeating everything twice and then meking factor levels the same...

# First add a column of zeros to test data to where our responce will be, then merge testing and training datasets.
TestData$Survived = rep(0,418)
df <- rbind(RawData, TestData)


# Deleting PassengerId, cabin and Ticket columns
df<-df[,-c(1,9,11)]


# Engeneering Title feature from Name:
df$Name <- as.character(df$Name)
df$Title <- sapply(df$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
# Drop spaces
df$Title <- sub(' ', '', df$Title)
# Unite categories wilh small amount of members
df$Title[df$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
df$Title[df$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
df$Title[df$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

df$FamilySize <- df$SibSp + df$Parch + 1


# FamilyID: large particular family or small.
df$Surname <- sapply(df$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
df$FamilyID <- paste(as.character(df$FamilySize), df$Surname, sep="")
df$FamilyID[df$FamilySize <= 2] <- 'Small'

famIDs <- data.frame(table(df$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
df$FamilyID[df$FamilyID %in% famIDs$Var1] <- 'Small'

df$FamilyID2 <- df$FamilyID
df$FamilyID2 <- as.character(df$FamilyID2)
df$FamilyID2[df$FamilySize <= 3] <- 'Small'
df$FamilyID2 <- factor(df$FamilyID2)


#######################
#######################


# There are many NAs in Age, as well as several in Embarked and fare. Let's impute them:

# Imputing age
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=df[!is.na(df$Age),], method="anova")
df$Age[is.na(df$Age)] <- predict(Agefit, df[is.na(df$Age),])

# Imputing Embarked. Only one NA here, impute value of the majority.
df$Embarked[c(62,830)] = "S"
df$Embarked <- factor(df$Embarked)

# Imputing fare
df$Fare[1044] <- median(df$Fare, na.rm=TRUE)


# Treat Title, Survived, Pclass and FamilyID as cathegorical variables.
df$FamilyID <- factor(df$FamilyID)
df$Title <- factor(df$Title)
df$Survived <- factor(df$Survived)
df$Pclass <- factor(df$Pclass)

# Separate processed data back into training and testing sets.
RawData = df[1:891,]
TestData = df[892:1309,]

# Save all the data
save(file="TestData.rda", x=TestData)
save(file="RawData.rda", x=RawData)

