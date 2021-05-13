#Importing dataset

#dataset = read.csv('data/Telco-Customer-Churn.csv')
setwd(getwd())
dataset = read.csv('Telco-Customer-Churn.csv')

# Setting "No" values 
library(plyr)
dataset[dataset == 'No internet service'] <- 'No'
dataset[dataset == 'No phone service'] <- 'No'
dataset$SeniorCitizen[dataset$SeniorCitizen == 1] <- 'is Senior Citizen'
dataset$SeniorCitizen[dataset$SeniorCitizen == 0] <- 'is not Senior Citizen'
dataset$Partner <- revalue(dataset$Partner, c('Yes' = 'has Partner', 'No' = 'Single'))
dataset$Dependents <- revalue(dataset$Dependents, c('Yes' = 'has Dependents', 'No' = 'no Dependents'))
dataset$InternetService <- revalue(dataset$InternetService, c('No' = 'No Internet'))
dataset$PhoneServ <- paste(dataset$PhoneService, dataset$MultipleLines)
dataset$PhoneServ <- revalue(dataset$PhoneServ, c('Yes Yes' = 'Multiple Lines', 'Yes No' = 'One Line', 'No No' = 'No Phone'))
dataset$Streaming<- paste(dataset$StreamingTV, dataset$StreamingMovies)
dataset$Streaming <- revalue(dataset$Streaming, c('Yes Yes' = 'TV & Movies', 'Yes No' = 'Only TV', 'No Yes' = 'Only Movies', 'No No' = 'No Streaming'))

# Adding Tenure Categories
library(dplyr)
dataset <- (mutate(dataset, tenure_group = ifelse(dataset$tenure %in% 0:12, "0-12 Months",
                                                  ifelse(dataset$tenure %in% 13:24, "13-24 Months",
                                                         ifelse(dataset$tenure %in% 25:36, "25-36 Months",
                                                                ifelse(dataset$tenure %in% 37:48, "37-48 Months",
                                                                       ifelse(dataset$tenure %in% 49:60, "49-60 Months","over 60 Months")))))))

dataset$tenure_group <- as.factor(dataset$tenure_group)

# Setting Churn Values
dataset$Churn = as.character(dataset$Churn)
dataset$Churn[dataset$Churn == 'No'] <- 'Stayed'
dataset$Churn[dataset$Churn == 'Yes'] <- 'Churn'


# # other data types
dataset$gender = as.character(dataset$gender)
dataset$Dependents = as.character(dataset$Dependents)
dataset$Partner = as.character(dataset$Partner)

# separating churn and non-churn customers
churn <- filter(dataset, Churn == 'Churn')
non_churn <- filter(dataset, Churn == 'Stayed')


str(dataset)

temp_tenure <- dataset$tenure

dataset$tenure <- as.numeric(dataset$tenure)
dataset$MonthlyCharges <- as.numeric(dataset$MonthlyCharges)
dataset$TotalCharges <- as.numeric(dataset$TotalCharges)

s <- c("customerID","gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines",
       "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
       "StreamingMovies","Contract","PaperlessBilling","PaymentMethod","Churn","PhoneServ","Streaming")

for (i in 1:length(s)){
  dataset[,s[i]] <- as.factor(dataset[,s[i]])
}

str(dataset)

rm(i,s)


#applying predictive modeling using h2o

library(h2o)
h2o.init(nthreads=12, max_mem_size="64g")

data <- as.h2o(dataset)

data1 <- data[,-1]
head(data1)

#splitting into train & test data

y <- "Churn"                                
x <- setdiff(names(data1), y)              
parts <- h2o.splitFrame(data1, 0.8, seed=99)
train <- parts[[1]]                         
test <- parts[[2]]  

#applying models

m1 <- h2o.deeplearning(x, y, train)
m2 <- h2o.randomForest(x, y, train)
m3 <- h2o.gbm(x, y, train)

h2o.performance(m1,test)
h2o.performance(m1,train)

h2o.performance(m2,test)
h2o.performance(m2,train)

h2o.performance(m3,test)
h2o.performance(m3,train)

h2o.confusionMatrix(m1, test)
h2o.confusionMatrix(m2, test)
h2o.confusionMatrix(m3, test)


var_imp1 <- h2o.varimp(m1)
head(var_imp1,n=5)
var_imp2 <- h2o.varimp(m2)
head(var_imp2,n=5)
var_imp3 <- h2o.varimp(m3)
head(var_imp3,n=5)
library(ggplot2)


#confusion matrix for deeplearning
TClass <- factor(c(0, 0, 1, 1))
PClass <- factor(c(0, 1, 0, 1))
Y <- c(957,69,191,181)
df <- data.frame (TClass, PClass, Y)

ggplot(data =  df, mapping = aes(x = TClass, y = PClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_bw() + theme(legend.position = "none")

#confusion matrix for random forest
TClass <- factor(c(0, 0, 1, 1))
PClass <- factor(c(0, 1, 0, 1))
Y <- c(945,81,179,193)
df1 <- data.frame (TClass, PClass, Y)

ggplot(data =  df1, mapping = aes(x = TClass, y = PClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_bw() + theme(legend.position = "none")

#confusion matrix for gbm
TClass <- factor(c(0, 0, 1, 1))
PClass <- factor(c(0, 1, 0, 1))
Y <- c(924,102,150,222)
df2 <- data.frame (TClass, PClass, Y)

ggplot(data =  df2, mapping = aes(x = TClass, y = PClass)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_bw() + theme(legend.position = "none")

