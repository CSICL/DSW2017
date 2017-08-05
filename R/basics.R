#Hello World
print('Hello World!')

# R as a calculator
4*2

# creating vectors
myVec0 <- c(1,2,3,4)
myVec0
myVec0[3]

myVec1 <- c('Cat', 'Dog', 'Mouse', "Chicken")


myVec2 <- seq(1:20)
myVec2

#creating matrices
myMat1 <- matrix(myVec2, ncol=4)
myMat1
myMat1[1,3]
myMat1[,3]
myMat1[1,]

#Lets roll some dice
sample(1:6, 2)

#R loops
for (i in 1:10) print(i^2)


