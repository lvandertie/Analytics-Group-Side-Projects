# Data Nerds
# Digit Recognizer R Code

# AT A HIGH LEVEL
# Computer vision is when code looks at an image as a set of values for each pixel, treating visual 
# items as a matrix (converted to a vector/row by unpivoting the matrix)

# From Kaggle on this data:
# Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. 
# Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of 
# that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, 
# inclusive.

# Load libraries
# Install package by running command: install.packages("readr")
library(readr)
library(caret)

# Set working directory so R knows where to look for files
setwd("C:/Users/baseb/Google Drive/Math/Independent Analytics - Software + Advanced Stats/Kaggle/Digit Recognizer/all")
# Read in csv's
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# Check dimensions of the dataframe
dim(train)

# Look at distribution of the column "Label"
#   Note: use the function as.factor() because we want to treat as categorical not numerical
#     If we used table() on numerical values it would return distribution metrics like mean, median, quartiles, etc
table(as.factor(train$label))

# Plot the distribution of digits with cool coloring
ggplot(train,aes(x=as.factor(label),fill=label))+
  geom_bar(stat="count",color="white")+
  scale_fill_gradient(low="lightblue",high="pink",guide=FALSE)+
  labs(title="Digits in Train Data",x="Digits")

# Pick a sample of 50 row numbers
sample <- sample(1:nrow(train),50)

# Select the subset of the training data set and remove the first column.
# Then, apply t() to transpose (need to do this to fit the format of image() later on)
var <- t(train[sample,-1])
# Convert row of grayscale values to matrices
var_matrix <- lapply(1:50,function(x) matrix(var[,x],ncol=28))
# Set some graphcial parameters relating to margin size
opar <- par(no.readonly = T)
par(mfrow=c(5,10),mar=c(.1,.1,.1,.1))
# For each of the samples, reverse/revolve the matrix then run the image() function.
#   Rev() used to make sure image is oriented correctly 
for(i in 1:50) {
  for(j in 1:28) {
    var_matrix[[i]][j,] <- rev(var_matrix[[i]][j,])
  }
  image(var_matrix[[i]],col=grey.colors(225),axes=F)
}

# ---- Data Processing ----

# Checking if there are any predictors (pixel locations) that can be removed because they add no value
# nearZeroVar() checks what predictors either have:
#   - All identical values (typically zeroes in this example)
#   - very few distinct values, and the most common value is still significantly 
#     more frequent that the second most common, essentially just one value
nzr <- nearZeroVar(train[,-1],saveMetrics=T,freqCut=10000/1,uniqueCut=1/7)
# Check how many predictors can get removed (probably the pixels in the corners of 
# images because almost always are the gray/black background)
sum(nzr$zeroVar)
sum(nzr$nzv)
# Remove useless predictors as identified by nearZeroVar()
cutvar <- rownames(nzr[nzr$nzv==TRUE,])
var <- setdiff(names(train),cutvar)
train <- train[,var]

# ---- PCA ----
# PCA: Principal Component Analysis is a dimensionality reduction technique to map higher 
# dimension (# of predictors) data into lower dimensions (# predictors) to simplify the models
# Attempts to reduce dimensions while retaining as much variance as possible between observations
# First principal component will retain the most variance while the second will retain the second most, etc
# Set aside label column and then drop from train data frame
label <- as.factor(train[[1]])
train$label <- NULL
train <- train/255
# Create covariance matrix used in PCA
covtrain <- cov(train)
# Run PCA with prcomp()
train_pc <- prcomp(covtrain)
# Calculate variance retained by each PC
varex <- train_pc$sdev^2/sum(train_pc$sdev^2)
# Calculate running sum of variance retained/explained by PC's
varcum <- cumsum(varex)
result <- data.frame(num=1:length(train_pc$sdev),
                     ex=varex,
                     cum=varcum)
# View running sum of variance retained as # of PC's increases
plot(result$num,result$cum,type="b",xlim=c(0,100),
     main="Variance Explained by Top 100 Components",
     xlab="Number of Components",ylab="Variance Explained")
# Drop a line at 25th PC because that is about when adding additional PC's is no longer helpful
# In general: look for elbow of the graph
abline(v=25,lty=2)

# Set up matrix of predicted values by multiplying the matrix of values by Principal Components
# This is matrix-vector multiplication as Principal Components are actually eigenvectors of the
# covariance (I believe covariance) matrix.  Projections are original values times PC vector
train_score <- as.matrix(train) %*% train_pc$rotation[,1:25]
train <- cbind(label,as.data.frame(train_score))

# Plot projections using first two Principal Components to see how well using just two PC's can segement the projections
colors <- rainbow(length(unique(train$label)))
names(colors) <- unique(train$label)
plot(train$PC1,train$PC2,type="n",main="First Two Principal Components")
text(train$PC1,train$PC2,label=train$label,col=colors[train$label])
