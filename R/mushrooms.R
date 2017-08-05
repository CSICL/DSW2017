#evoke packages
#caret & radonForest
library(randomForest)
library(caret)

#import data set to R
data <- read.csv('~/NEW/work/AugustWorkshop/Scripts/MushroomData.csv')
#data <- read.csv('~/R/HandsOn/MushroomData.csv')

#have a look at the data
View(data)
head(data)
summary(data)

#Create data for training
sample.ind <- sample(c(1,2), nrow(data), replace = T,
                    prob = c(0.7, 0.3))

data.train <- data[sample.ind==1,]
data.test <- data[sample.ind==2,]

#See how data sets look as edible vs poisonous
table(data$Edible)/nrow(data)
table(data.train$Edible)/nrow(data.train)
table(data.test$Edible)/nrow(data.test)


#run the random forest classification algorithm
rf <- randomForest(Edible ~ ., data = data.train)
print(rf)

#spot the important features
varImpPlot(rf,
           sort = T,
           n.var=10,
           main="Top 10 - Variable Importance")


var.imp <- data.frame(importance(rf, type=2))

#View(var.imp)

#predict the test data set


data.test$predicted.response <- predict(rf, data.test)
confusionMatrix(data=data.test$predicted.response, reference=data.test$Edible)


#Accuracy = (True Positive+True Negative)/(Total)



