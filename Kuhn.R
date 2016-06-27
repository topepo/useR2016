###################################################################
## Code for the useR 2016 tutorial "Never Tell Me the Odds! Machine 
## Learning with Class Imbalances" by Max Kuhn
## 
## Slides and this code can be found at
##    https://github.com/topepo/useR2016
## 
## packages used here are: caret, pROC, rpart, partykit, randomForest,
##   AppliedPredictiveModeling, DMwR, ROSE, C50, kernlab, ggthemes,
##   plyr
## 
## Session info is at the bottom of this document
## 
## Data are at: https://github.com/rudeboybert/JSE_OkCupid
##              https://github.com/topepo/useR2016
##
## OkC data are created in the file okc_data.R in the useR2016 repo
##


###################################################################
## Create toy data used throught the slides

library(AppliedPredictiveModeling)
set.seed(14034)
ex_dat <- easyBoundaryFunc(250, intercept = -6, interaction = 1.5)

library(ggplot2)
ggplot(ex_dat, aes(x = X1, y = X2)) + 
  geom_point(aes(color = class), cex = 3, alpha = .5) + 
  theme(legend.position = "top") + 
  scale_colour_tableau()  + 
  xlab("Predictor A") + ylab("Predictor B")

###################################################################
## Slide 22 "Example Data - Electronic Medical Records"

load("emr.RData")

str(emr, list.len = 20)

###################################################################
## Slide 23 "Example Data - Electronic Medical Records"

library(caret)

set.seed(1732)
emr_ind <- createDataPartition(emr$Class, p = 2/3, list = FALSE)
emr_train <- emr[ emr_ind,]
emr_test  <- emr[-emr_ind,]

mean(emr_train$Class == "event")
mean(emr_test$Class == "event")

table(emr_train$Class)
table(emr_test$Class)

###################################################################
## Slide 25 "Example Data - OKCupid"

load("okc.RData") ## create this using the file "okc_data.R"
str(okc, list.len = 20, vec.len = 2)

###################################################################
## Slide 26 "Example Data - OKCupid"

set.seed(1732)
okc_ind <- createDataPartition(okc$Class, p = 2/3, list = FALSE)
okc_train <- okc[ okc_ind,]
okc_test  <- okc[-okc_ind,]

mean(okc_train$Class == "stem")
mean(okc_test$Class == "stem")

###################################################################
## Slide 40 and 43 "A Single Shallow Tree"

library(rpart)
library(partykit)
rp1 <- rpart(Class ~ ., data = emr_train, control = rpart.control(maxdepth = 3, cp = 0))
plot(as.party(rp1))

###################################################################
## Slide 44 "A Single Shallow Tree (Bootstrapped)"

set.seed(9595)
dat2 <- emr_train[sample(1:nrow(emr_train), nrow(emr_train), replace = TRUE),]
rp2 <- rpart(Class ~ ., data = dat2, control = rpart.control(maxdepth = 3, cp = 0))
plot(as.party(rp2))

###################################################################
## Slide 45 "A Single Shallow Tree (Bootstrapped)"

set.seed(1976)
dat3 <- emr_train[sample(1:nrow(emr_train), nrow(emr_train), replace = TRUE),]
rp3 <- rpart(Class ~ ., data = dat3, control = rpart.control(maxdepth = 3, cp = 0))
plot(as.party(rp3))

###################################################################
## Slide 47 "Random Forests with the EMR Data"

## on OS X, I ran in parallel using 
## library(doMC)
## registerDoMC(cores=8)
## on Windows, try the doParallel package
## **if** your computer has multiple cores and sufficient memory

ctrl <- trainControl(method = "repeatedcv",
                     repeats = 5, 
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = twoClassSummary)
emr_grid <- data.frame(mtry = c(1:15, (4:9)*5))

set.seed(1537)
rf_emr_mod <- train(Class ~ ., 
                    data = emr_train,
                    method = "rf",
                    metric = "ROC",
                    tuneGrid = emr_grid,
                    ntree = 1000,
                    trControl = ctrl)

###################################################################
## Back to Slide 37 "ROC Curve" to plot the **test set data**

exRoc <- roc(emr_test$Class, ex_probs$Prob, levels = rev(levels(emr_test$Class)))
plot(exRoc, legacy.axes = FALSE,
     print.thres=c(.2, .5,  1), 
     print.thres.pattern = "%.2f (Sp = %.3f, Sn = %.3f)",
     print.thres.cex = .8)

###################################################################
## Slide 50 "Random Forest Results - EMR Example"

ggplot(rf_emr_mod)

###################################################################
## Slide 51 "Approximate Random Forest Resampled ROC Curve"

## This function averages the class probability values per sample
## across the hold-outs to get an averaged ROC curve

roc_train <- function(object, best_only = TRUE, ...) {
  library("pROC")
  library("plyr")
  
  if(object$modelType != "Classification")
    stop("ROC curves are only available for classification models")
  if(!any(names(object$modelInfo) == "levels"))
    stop(paste("The model's code is required to have a 'levels' module.",
               "See http://topepo.github.io/caret/custom_models.html#Components"))
  lvs <- object$modelInfo$levels(object$finalModel)
  if(length(lvs) != 2) 
    stop("ROC curves are only implemented here for two class problems")
  
  ## check for predictions
  if(is.null(object$pred)) 
    stop(paste("The out of sample predictions are required.",
               "See the `savePredictions` argument of `trainControl`"))
  
  if(best_only) {
    object$pred <- merge(object$pred, object$bestTune)
  }
  ## find tuning parameter names
  p_names <- as.character(object$modelInfo$parameters$parameter)
  p_combos <- object$pred[, p_names, drop = FALSE]
  
  ## average probabilities across resamples
  object$pred <- plyr::ddply(.data = object$pred, 
                             .variables = c("obs", "rowIndex", p_names),
                             .fun = function(dat, lvls = lvs) {
                               out <- mean(dat[, lvls[1]])
                               names(out) <- lvls[1]
                               out
                             })
  
  make_roc <- function(x, lvls = lvs, nms = NULL, ...) {
    out <- pROC::roc(response = x$obs,
                     predictor = x[, lvls[1]],
                     levels = rev(lvls))
    
    out$model_param <- x[1,nms,drop = FALSE]
    out
  }
  out <- plyr::dlply(.data = object$pred, 
                     .variables = p_names,
                     .fun = make_roc,
                     lvls = lvs,
                     nms = p_names)
  if(length(out) == 1)  out <- out[[1]]
  out
}

plot(roc_train(rf_emr_mod), 
     legacy.axes = TRUE,
     print.thres = .5,
     print.thres.pattern="   <- default %.1f threshold")

###################################################################
## Slide 52 "A Better Cutoff"

plot(roc_train(rf_emr_mod), 
     legacy.axes = TRUE,
     print.thres.pattern = "Cutoff: %.2f (Sp = %.2f, Sn = %.2f)",
     print.thres = "best")

###################################################################
## Slide 59 "Down-Sampling - EMR Data"

down_ctrl <- ctrl
down_ctrl$sampling <- "down"
set.seed(1537)
rf_emr_down <- train(Class ~ ., 
                     data = emr_train,
                     method = "rf",
                     metric = "ROC",
                     tuneGrid = emr_grid,
                     ntree = 1000,
                     trControl = down_ctrl)

###################################################################
## Slide 60 "Down-Sampling - EMR Data"

ggplot(rf_emr_down)

###################################################################
## Slide 61 "Approximate Resampled ROC Curve with Down-Sampling"

plot(roc_train(rf_emr_down), 
     legacy.axes = TRUE,
     print.thres = .5,
     print.thres.pattern="   <- default %.1f threshold")

###################################################################
## Slide 63 "Internal Down-Sampling - EMR Data"

set.seed(1537)
rf_emr_down_int <- train(Class ~ ., 
                         data = emr_train,
                         method = "rf",
                         metric = "ROC",
                         ntree = 1000,
                         tuneGrid = emr_grid,
                         trControl = ctrl,
                         ## These are passed to `randomForest`
                         strata = emr_train$Class,
                         sampsize = rep(sum(emr_train$Class == "event"), 2))

###################################################################
## Slide 64 "Internal Down-Sampling - EMR Data"

ggplot(rf_emr_down_int)

###################################################################
## Slide 67 "Up-Sampling - EMR Data"

up_ctrl <- ctrl
up_ctrl$sampling <- "up"
set.seed(1537)
rf_emr_up <- train(Class ~ ., 
                   data = emr_train,
                   method = "rf",
                   tuneGrid = emr_grid,
                   ntree = 1000,
                   metric = "ROC",
                   trControl = up_ctrl)

###################################################################
## Slide 68 "Up-Sampling - EMR Data"

ggplot(rf_emr_up)

###################################################################
## Slide 73 "SMOTE - EMR Data"

smote_ctrl <- ctrl
smote_ctrl$sampling <- "smote"
set.seed(1537)
rf_emr_smote <- train(Class ~ ., 
                      data = emr_train,
                      method = "rf",
                      tuneGrid = emr_grid,
                      ntree = 1000,
                      metric = "ROC",
                      trControl = smote_ctrl)

###################################################################
## Slide 74 "SMOTE - EMR Data"

ggplot(rf_emr_smote)

###################################################################
## Slide 75 "SMOTE - EMR Data"

emr_test_pred <- data.frame(Class = emr_test$Class)
emr_test_pred$normal <- predict(rf_emr_mod, emr_test, type = "prob")[, "event"]
emr_test_pred$down <- predict(rf_emr_down, emr_test, type = "prob")[, "event"]
emr_test_pred$down_int <- predict(rf_emr_down_int, emr_test, type = "prob")[, "event"]
emr_test_pred$up <- predict(rf_emr_up, emr_test, type = "prob")[, "event"]
emr_test_pred$smote <- predict(rf_emr_smote, emr_test, type = "prob")[, "event"]

get_auc <- function(pred, ref) auc(roc(ref, pred, levels = rev(levels(ref))))

apply(emr_test_pred[, -1], 2, get_auc, ref = emr_test_pred$Class)

###################################################################
## Slide 81 "CART and Costs - OkC Data"

fourStats <- function (data, lev = levels(data$obs), model = NULL) {
  accKapp <- postResample(data[, "pred"], data[, "obs"])
  out <- c(accKapp,
           sensitivity(data[, "pred"], data[, "obs"], lev[1]),
           specificity(data[, "pred"], data[, "obs"], lev[2]))
  names(out)[3:4] <- c("Sens", "Spec")
  out
}

ctrl_cost <- trainControl(method = "repeatedcv",
                          repeats = 5,
                          savePredictions = TRUE,
                          summaryFunction = fourStats)

###################################################################
## Slide 82 "CART and Costs - OkC Data"

## Get an initial grid of Cp values
rpart_init <- rpart(Class ~ ., data = okc_train, cp = 0)$cptable

cost_grid <- expand.grid(cp = rpart_init[, "CP"], Cost = 1:5)

## Use the non-formula method. Many of the predictors are factors and
## this will preserve the factor encoding instead of using dummy 
## variables. 

set.seed(1537)
rpart_costs <- train(x = okc_train[, names(okc_train) != "Class"],
                     y = okc_train$Class,
                     method = "rpartCost",
                     tuneGrid = cost_grid,
                     metric = "Kappa",
                     trControl = ctrl_cost)

###################################################################
## Slide 84 "CART and Costs - OkC Data"

ggplot(rpart_costs) + 
  scale_x_log10() + 
  theme(legend.position = "top")

###################################################################
## Slide 85 "CART and Costs - OkC Data"

ggplot(rpart_costs, metric = "Sens") + 
  scale_x_log10() + 
  theme(legend.position = "top")

###################################################################
## Slide 86 "CART and Costs - OkC Data"

ggplot(rpart_costs, metric = "Spec") + 
  scale_x_log10() + 
  theme(legend.position = "top")

###################################################################
## Slide 87 "C5.0 and Costs - OkC Data"

cost_grid <- expand.grid(trials = c(1:10, 20, 30),
                         winnow = FALSE, model = "tree",
                         cost = c(1, 5, 10, 15))
set.seed(1537)
c5_costs <- train(x = okc_train[, names(okc_train) != "Class"],
                  y = okc_train$Class,
                  method = "C5.0Cost",
                  tuneGrid = cost_grid,
                  metric = "Kappa",
                  trControl = ctrl_cost)

###################################################################
## Slide 89 "C5.0 and Costs - OkC Data"

ggplot(c5_costs) + theme(legend.position = "top")

###################################################################
## Slide 91 "OkC Test Results - C5.0"

rp_pred <- predict(rpart_costs, newdata = okc_test)
confusionMatrix(rp_pred, okc_test$Class)

###################################################################
## Slide 90 "OkC Test Results - CART"

c5_pred <- predict(c5_costs, newdata = okc_test)
confusionMatrix(c5_pred, okc_test$Class)

###################################################################
## Slide 103 "CART and Costs and Probabilities"

cost_mat <-matrix(c(0, 1, 5, 0), ncol = 2)
rownames(cost_mat) <- colnames(cost_mat) <- levels(okc_train$Class)
rp_mod <- rpart(Class ~ ., data = okc_train, parms = list(loss = cost_mat))
pred_1 <- predict(rp_mod, okc_test, type = "class")
pred_2 <- ifelse(predict(rp_mod, okc_test)[, "stem"] >= .5, "stem", "other")
pred_2 <- factor(pred_2, levels = levels(pred_1))

table(pred_1, pred_2)

###################################################################
## Session info:

# R Under development (unstable) (2016-06-07 r70726)
# Platform: x86_64-apple-darwin13.4.0 (64-bit)
# Running under: OS X 10.10.5 (Yosemite)
# 
# locale:
# [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
# 
# attached base packages:
# [1] parallel  grid      stats     graphics  grDevices utils     datasets
# [8] methods   base
# 
# other attached packages:
# [1] vcd_1.4-1                       ggthemes_3.0.3
# [3] kernlab_0.9-24                  RColorBrewer_1.1-2
# [5] randomForest_4.6-12             doMC_1.3.4
# [7] iterators_1.0.8                 foreach_1.4.3
# [9] inTrees_1.1                     C50_0.1.0-24
# [11] plyr_1.8.4                      ROSE_0.0-3
# [13] DMwR_0.4.1                      proxy_0.4-15
# [15] AppliedPredictiveModeling_1.1-6 partykit_1.0-5
# [17] rpart_4.1-10                    nnet_7.3-12
# [19] Hmisc_3.17-4                    Formula_1.2-1
# [21] survival_2.39-2                 caret_6.0-70
# [23] ggplot2_2.1.0                   lattice_0.20-33
# [25] pROC_1.8                        knitr_1.13
# 
# loaded via a namespace (and not attached):
# [1] splines_3.4.0       gtools_3.5.0        assertthat_0.1
# [4] TTR_0.23-1          highr_0.5.1         stats4_3.4.0
# [7] latticeExtra_0.6-28 arules_1.4-1        quantreg_5.21
# [10] chron_2.3-47        digest_0.6.9        minqa_1.2.4
# [13] RRF_1.6             colorspace_1.2-6    gbm_2.1.1
# [16] Matrix_1.2-6        SparseM_1.7         xtable_1.8-2
# [19] scales_0.4.0        gdata_2.17.0        lme4_1.1-12
# [22] MatrixModels_0.4-1  mgcv_1.8-12         car_2.1-2
# [25] ROCR_1.0-7          pbkrtest_0.4-6      quantmod_0.4-5
# [28] magrittr_1.5        evaluate_0.8.3      CORElearn_1.47.1
# [31] nlme_3.1-127        MASS_7.3-45         gplots_3.0.1
# [34] xts_0.9-7           foreign_0.8-66      class_7.3-14
# [37] tools_3.4.0         data.table_1.9.6    formatR_1.3
# [40] stringr_1.0.0       munsell_0.4.3       cluster_2.0.4
# [43] compiler_3.4.0      e1071_1.6-7         caTools_1.17.1
# [46] nloptr_1.0.4        bitops_1.0-6        labeling_0.3
# [49] gtable_0.2.0        codetools_0.2-14    abind_1.4-3
# [52] reshape2_1.4.1      gridExtra_2.2.1     zoo_1.7-12
# [55] KernSmooth_2.23-15  stringi_1.0-1       Rcpp_0.12.4
# [58] acepack_1.3-3.3     lmtest_0.9-34
