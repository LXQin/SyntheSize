##### Import libraries ####
library(ggplot2)
library(tidyverse)
library(DANA)
library(cowplot)
library(ggpubr)
library(ggsci)
library(cowplot)
library(reshape2)
library(glmnet)
library(e1071)
library(caret)
library(randomForest)
library(xgboost)
library(ROCR)
library(class)


#### Classification functions ####
# train_data, test_data are data frames with samples in rows.
# train_labels, test_labels are numbers with 0 for the first level, 1 for the second level.
# require train_data and test_data to be scaled and centered for successful genes
# LOGIS: 5 folds cv to choose penalty
# SVM: default
# KNN: choose k = 20, does not provides probabilities, auc = 0
# RF: ntree = 100
# XGB: nrounds = 25
# eval_classifier: evaluation is through 5 folds cv for each candidate sample size and each draw.

# Function for Logistic Regression with Lasso penalty
LOGIS <- function(train_data, train_labels, test_data, test_labels) {
  model <- cv.glmnet(as.matrix(train_data), as.factor(train_labels),
                     alpha = 0, family = "binomial", nfolds = 5)
  predictions <- predict(model, newx = as.matrix(test_data),
                         s = "lambda.min", type = "response")
  accuracy <- sum((predictions > 0.5) == test_labels) / length(test_labels)
  prediction_obj <- prediction(predictions, test_labels)
  auc <- performance(prediction_obj, measure = "auc")@y.values[[1]]
  res <- c(accuracy, auc)
  names(res) <- c("accuracy","auc")
  return(res)
}

# Function for SVM
SVM <- function(train_data, train_labels, test_data, test_labels) {
  model <- svm(as.factor(train_labels) ~ ., data = train_data,
               probability = TRUE)
  predictions <- predict(model, test_data, probability=TRUE)
  predictions <- attr(predictions, "probabilities")[,2]
  accuracy <- sum((predictions > 0.5) == test_labels) / length(test_labels)
  prediction_obj <- prediction(predictions, test_labels)
  auc <- performance(prediction_obj, measure = "auc")@y.values[[1]]
  res <- c(accuracy, auc)
  names(res) <- c("accuracy","auc")
  return(res)
}

# Function for k-Nearest Neighbors (k-NN)
KNN <- function(train_data, train_labels, test_data, test_labels) {
  model <- knn(train = train_data, test = test_data,
               cl = as.factor(train_labels), k = 20)
  accuracy <- sum(model == test_labels) / length(test_labels)
  auc <- 0
  res <- c(accuracy, auc)
  names(res) <- c("accuracy","auc")
  return(res)
}

# Function for Random Forest
RF <- function(train_data, train_labels, test_data, test_labels) {
  model <- randomForest(x = train_data, y = as.factor(train_labels),
                        xtest = test_data, ytest = as.factor(test_labels),
                        ntree = 100, keep.forest = TRUE) 
  predictions <- predict(model, test_data, type="prob")
  accuracy <- sum(predict(model, test_data) == test_labels) / length(test_labels)
  prediction_obj <- prediction(predictions[,2], test_labels)
  auc <- performance(prediction_obj, measure = "auc")@y.values[[1]]
  res <- c(accuracy, auc)
  names(res) <- c("accuracy","auc")
  return(res)
}

# Function for XGBoost
XGB <- function(train_data, train_labels, test_data, test_labels) {
  model <- xgboost(data = as.matrix(train_data), label = train_labels,
                   nrounds = 25, verbose = 0, 
                   objective = "binary:logistic")
  
  predictions <- predict(model, as.matrix(test_data))
  accuracy <- sum((predictions > 0.5) == test_labels) / length(test_labels)
  prediction_obj <- prediction(predictions, test_labels)
  auc <- performance(prediction_obj, measure = "auc")@y.values[[1]]
  res <- c(accuracy, auc)
  names(res) <- c("accuracy","auc")
  return(res)
}

eval_classifier <- function(whole_generated, whole_groups = NULL,
                            n_candidate, n_draw = 5, log = TRUE){
  # This function perform multiple classification algorithm on the candidate sample size
  # @params: whole_generated - generated data matrix with samples in rows
  # @params: whole_groups - vector containing the group information of samples
  # @params: log - whether the data is log2 transformed
  # @params: n_candidate - the candidate total sample sizes, half of them for each group label
  # @params: n_draw - the number of times drawing n_candidate from the whole_generated
  
  if(!log){
    whole_generated <- log2(whole_generated + 1)
  }
  g1 <- unique(whole_groups)[1]
  g2 <- unique(whole_groups)[2]
  dat_g1 <- whole_generated[whole_groups == g1, ]
  dat_g2 <- whole_generated[whole_groups == g2, ]
  
  # if(failure == "remove"){
  #    dat_g1 <- dat_g1[, apply(dat_g1, 2, sd) != 0]
  #    dat_g2 <- dat_g2[, apply(dat_g2, 2, sd) != 0]
  # }
  res <- data.frame(total_size = numeric(0), draw = numeric(0), 
                    method = character(0), accuracy = numeric(0), 
                    auc = numeric(0))
  
  for (n_index in 1:length(n_candidate)) {
    n <- n_candidate[n_index]
    cat(n_index)
    for (draw in 1:n_draw) {
      cat(draw)
      dat_g1_candidate <- dat_g1[sample(1:nrow(dat_g1), n/2, replace = F),]
      dat_g2_candidate <- dat_g2[sample(1:nrow(dat_g2), n/2, replace = F),]
      dat_candidate <- rbind(dat_g1_candidate, dat_g2_candidate)
      groups_candidate <- rep(c(g1, g2), each = n/2)
      
      num_folds <- 5  
      df_acc <- data.frame(LOGIS = numeric(0), SVM = numeric(0), 
                           KNN = numeric(0), RF = numeric(0),
                           XGB = numeric(0))
      df_auc <- df_acc
      
      # Create stratified folds for cross-validation
      folds <- createFolds(groups_candidate, k = num_folds, 
                           list = TRUE, returnTrain = FALSE)
      # Iterate through the folds to get test and train data
      for (fold_index in 1:num_folds) {
        # Get the indices of samples in the current test fold
        test_indices <- folds[[fold_index]]
        
        # Get the test data and labels using the test indices
        test_data <- dat_candidate[test_indices,]
        test_labels <- ifelse(groups_candidate[test_indices] == g1, 0, 1)
        
        # Get the indices of samples in the training fold
        train_indices <- setdiff(1:nrow(dat_candidate), test_indices)
        
        # Get the training data and labels using the training indices
        train_data <- dat_candidate[train_indices,]
        train_labels <- ifelse(groups_candidate[train_indices] == g1, 0, 1)
        
        # Scaling for successful features
        train_data[,apply(train_data, 2, sd) != 0] <- scale(train_data[,apply(train_data, 2, sd) != 0])
        test_data[,apply(test_data, 2, sd) != 0] <- scale(test_data[,apply(test_data, 2, sd) != 0])
        
        # Classification
        fit_logis <- LOGIS(train_data, train_labels, test_data, test_labels)
        fit_svm <- SVM(train_data, train_labels, test_data, test_labels)
        fit_knn <- KNN(train_data, train_labels, test_data, test_labels)
        fit_rf <- RF(train_data, train_labels, test_data, test_labels)
        fit_xgb <- XGB(train_data, train_labels, test_data, test_labels)
        
        df_acc[fold_index,] <- c(fit_logis["accuracy"], fit_svm["accuracy"],
                                 fit_knn["accuracy"], fit_rf["accuracy"],
                                 fit_xgb["accuracy"])
        df_auc[fold_index,] <- c(fit_logis["auc"], fit_svm["auc"],
                                 fit_knn["auc"], fit_rf["auc"],
                                 fit_xgb["auc"])    
        
      }
      res <- rbind(res, data.frame(total_size = n,
                                   draw = draw,
                                   method = c("LOGIS", "SVM", "KNN",
                                              "RF","XGB"),
                                   accuracy = apply(df_acc,2,mean),
                                   auc = apply(df_auc, 2, mean)))
    }
  }
  return(res)
}

#### Evaluation functions #### 
heatmap_eval <-  function(dat_generated, dat_real){
  # This function evaluate the generated samples by looking at the heatmap
  # @params: dat_generated - the log2 data matrix with generated samples in rows, features in columns
  # @params: dat_real - the log2 data matrix with real samples in rows, features in columns
    value_generated <- data.frame(column = rep(1:ncol(dat_generated), rep(nrow(dat_generated), ncol(dat_generated))),
                                  row = rep(1:nrow(dat_generated), ncol(dat_generated)),
                                  value = (c(as.matrix(dat_generated))))
    value_real <- data.frame(column = rep(1:ncol(dat_real), rep(nrow(dat_real), ncol(dat_real))),
                             row = rep(1:nrow(dat_real), ncol(dat_real)),
                             value = (c(as.matrix(dat_real))))
    value_combine <- rbind(value_generated, value_real)
    value_combine$type <-  factor(rep(c("Generated", "Real"), c(nrow(value_generated), nrow(value_real))), levels = c("Real","Generated"))
    p_heat <- ggplot(data = value_combine, aes(x = column, y = row, fill = value))+
      geom_tile(show.legend = F)+
      facet_wrap(vars(type))+
      theme_bw()+
      labs(x = "Genes", y = "Samples")+
      theme(strip.text = element_text(face = "bold", size = rel(0.8)),
            strip.background = element_blank(),
            panel.border = element_rect(colour = "black", fill = NA),
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            axis.line = element_blank())
  return(p_heat)
}
UMAP_eval <- function(dat_generated, dat_real, groups_generated, groups_real, legend_pos = "top"){
  # This function perform UMAP dimension reduction to real and generated combined dataset 
  # and check whether the main variation lies between real and generated samples
  # @params: dat_generated - the log2 data matrix with generated samples in rows, features in columns
  # @params: dat_real - the log2 data matrix with real samples in rows, features in columns
  # @params: groups_generated - groups information for generated samples, samples from different groups are represented by different point shape
  # @params: groups_real - groups information for real samples, samples from different groups are represented by different point shape
  # if no group, groups_generated and groups_real should be only one same value.
  # failure genes (zero in all generated samples) in generated samples will be removed 
  
  dat_real <- dat_real[, apply(dat_generated, 2, sd) != 0]
  dat_generated <- dat_generated[, apply(dat_generated, 2, sd) != 0]
  datatype <- rep(c("Real","Generated"), c(nrow(dat_real), nrow(dat_generated)))
  groups_combine <- c(groups_real, groups_generated)
  mat <- as.matrix(rbind(dat_real, dat_generated))
  
  UMAP_fit = umap::umap(mat)
  UMAP_df = UMAP_fit$layout %>% 
    as.data.frame() %>%
    dplyr::rename(UMAP1 = "V1", UMAP2 = "V2") %>%
    mutate(ID = row_number())
  factor_df <- data.frame(groups = groups_combine, datatype = datatype) %>% mutate(ID=row_number())
  plot_df <- UMAP_df %>% inner_join(factor_df, by="ID")
  p_umap <- plot_df %>% ggplot(aes(x = UMAP1, y = UMAP2, color = datatype, shape = groups))+
                                geom_point()+
                                labs(color = "Datatype", shape = "Groups")+
                                theme_bw()+
                                scale_color_rickandmorty()+
                                theme(
                                  strip.text = element_text(face = "bold", size = rel(0.8)),
                                  strip.background = element_blank(),
                                  panel.border = element_rect(colour = "black", fill = NA),
                                  axis.text.x = element_blank(),
                                  axis.ticks.x = element_blank(),
                                  axis.ticks.y = element_blank(),
                                  axis.text.y = element_blank(),
                                  panel.grid.major = element_blank(),
                                  panel.grid.minor = element_blank(),
                                  axis.line = element_blank())+
                                guides(color = guide_legend(order=1, nrow=2),
                                       shape = guide_legend(order=2, nrow=2))+
                                theme(legend.position=legend_pos,
                                      legend.title = element_text(size=10),
                                      legend.text = element_text(size=10) )+
                  scale_color_aaas()
    
  return(p_umap)
}

curve_fit <- function(acc_table, acc_target = NULL, n_target = NULL, plot = TRUE, annotation = c("Accuracy", "")){
  # This is a function fitting power law curve between accuracy and sample size
  # @params acc_table - a data frame including two columns, n and accuracy
  # @params acc_target - the target accuracy used to compute sample size requirement
  # @params n_target - the target sample size used to compute accuracy
  # @params annotation - a vector including the ylabel, and main title
  # from paper: Predicting sample size required for classification performance Figuora 2012
  
  # specify model
  gradientF <- deriv3( ~(1 - a) - (b * (x^c)), c("a", "b", "c"), function(a, b, c, x) NULL)
  
  # fitting the model using nls
  m <- nls(y ~ gradientF(a, b, c, x), start = list(a = 0, b = 1, c = -0.5), 
           weights = seq(1, nrow(acc_table))/nrow(acc_table), 
           control = list(maxiter = 1000, warnOnly = TRUE),
           algorithm = "port", upper = list(a = 10, b = 10, c = -0.1), 
           lower = list(a = 0, b = 0, c=-10),
           data = data.frame(y = acc_table$accuracy, x = acc_table$n))
  
  
  # build fitted values and CI
  est <- fitted.values(m)
  if(var(est) == 0){
    cat("All estimated values are the same, cannot compute the CI")
    est_df <- data.frame(n = acc_table$n,
                         Estimated = acc_table$accuracy,
                         Predict = NA,
                         Accuracy = est,
                         Low = NA,
                         High = NA)
  }
  else{
    se_fit_est <- sqrt(apply(attr(predict(m, list(x = acc_table$n)),"gradient"), 1, function(x) sum(vcov(m)*outer(x,x))))
    est_ci <- est + outer(se_fit_est,qnorm(c(.5, .025,.975)))
    est_df <- data.frame(n = acc_table$n, 
                         Estimated = acc_table$accuracy,
                         Predict = NA,
                         Accuracy = est_ci[,1],
                         Low = est_ci[,2],
                         High = est_ci[,3])
  }
  
  # prediction and CI
  if(!is.null(acc_target)){
    cef <- coefficients(m)
    a <- cef[1]
    b <- cef[2]
    c <- cef[3]
    ss <- ((1-a-acc_target)/b)^(1/c)
  }
  if(!is.null(n_target)){
    prediction <- predict(m, list(x = n_target))
    if(var(est) != 0){
      se_fit <- sqrt(apply(attr(predict(m, list(x = n_target)),"gradient"), 1, function(x) sum(vcov(m)*outer(x,x))))
      prediction_ci <- prediction + outer(se_fit,qnorm(c(.5, .025,.975)))
      pre_df <- data.frame(n = n_target,
                           Estimated = NA, 
                           Predict = prediction_ci[,1],
                           Accuracy = prediction_ci[,1],
                           Low = prediction_ci[,2],
                           High =  prediction_ci[,3])
    }
    else{
      pre_df <- data.frame(n = n_target,
                           Estimated = NA, 
                           Predict = prediction,
                           Accuracy = prediction,
                           Low = NA,
                           High =  NA)
    } 
  }
  if(plot){
    df <- as.data.frame(rbind(est_df, pre_df))
    p <- ggplot(data = df)+
      geom_point(aes(x = n, y = Estimated, color = "Estimated"), size=2, show.legend = F)+
      geom_line(aes(x = n, y = Accuracy, color = "Fitted"), size=1, show.legend = T)+
      geom_point(aes(x = n, y = Predict, color = "Predicted"), show.legend = F)+
      labs(x = "Sample size", y = annotation[1], title = annotation[2])+
      scale_color_manual(name = "Type", breaks = c("Estimated","Fitted","Predicted"), 
                         values = c("Estimated" = "black", "Predicted" = "red", "Fitted" = "blue")) + 
      theme(panel.border = element_blank(), panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(), axis.line = element_blank())+
      theme_bw()+
      theme(legend.title=element_blank())+
      theme(legend.position = c(0.7, 0.28))+
      theme(plot.title = element_text(hjust = 0.5))+
      ylim(0, 1)
    if(var(est) != 0){
      p <- p + geom_ribbon(aes(x = n, ymin = Low, ymax = High), alpha = 0.2)
    }
  }
  return(list(graph = p, prediction = pre_df[,c("n", "Predict", "Low", "High")], model_fitted = m))
}

#### Visualization of the mean accuracy vs sample size ####
vis_classifier <- function(metric_generated, metric_real, n_target){
  
  mean_acc_generated <- metric_generated %>% group_by(total_size, method) %>%   summarise(accuracy = mean(accuracy))
  mean_acc_generated$n <- mean_acc_generated$total_size
  
  mean_auc_generated <- metric_generated %>% group_by(total_size, method) %>%   summarise(accuracy = mean(auc))
  mean_auc_generated$n <- mean_auc_generated$total_size
  
  mean_acc_real <- metric_real %>% group_by(total_size, method) %>% summarise(accuracy = mean(accuracy))
  mean_acc_real$n <- mean_acc_real$total_size
  
  mean_auc_real <- metric_real %>% group_by(total_size, method) %>% summarise(accuracy = mean(auc))
  mean_auc_real$n <- mean_auc_real$total_size
  
  acc_graphs <- list()
  auc_graphs <- list()
  for (classifier in unique(metric_real$method)) {
    ss_est_real <- curve_fit(acc_table = mean_acc_real[mean_acc_real$method == classifier,], n_target = n_target, annotation = c("Accuracy", paste(classifier,": TCGA", sep = "")))
    ss_est_generated <- curve_fit(acc_table = mean_acc_generated[mean_acc_generated$method == classifier,],  n_target = n_target, annotation = c("Accuracy", paste(classifier,": Generated", sep ="")))
    acc_graphs <- append(acc_graphs, list(ss_est_real$graph+ylim(0.5,0.85)))
    acc_graphs <- append(acc_graphs, list(ss_est_generated$graph+ylim(0.5,0.85)))
    if(classifier != "KNN"){
      ss_est_real <- curve_fit(acc_table = mean_auc_real[mean_auc_real$method == classifier,], n_target = n_target, annotation = c("AUC", paste(classifier,": TCGA", sep = "")))
      ss_est_generated <- curve_fit(acc_table = mean_auc_generated[mean_auc_generated$method == classifier,],  n_target = n_target, annotation = c("AUC", paste(classifier,": Generated", sep ="")))
      auc_graphs <- append(auc_graphs, list(ss_est_real$graph+ylim(0.72,0.9)))
      auc_graphs <- append(auc_graphs, list(ss_est_generated$graph+ylim(0.72,0.9)))
    }
  }
 
  print(ggarrange(acc_graphs[[1]], acc_graphs[[2]],
                  acc_graphs[[3]], acc_graphs[[4]],
                  acc_graphs[[5]], acc_graphs[[6]],
                  acc_graphs[[7]], acc_graphs[[8]],
                  acc_graphs[[9]], acc_graphs[[10]],
                  nrow = 5, ncol = 2, common.legend = T, legend = "bottom"))
}
vis_classifier_single <- function(metric_generated, n_target){
  
  mean_acc_generated <- metric_generated %>% group_by(total_size, method) %>%   summarise(accuracy = mean(accuracy))
  mean_acc_generated$n <- mean_acc_generated$total_size
  
  mean_auc_generated <- metric_generated %>% group_by(total_size, method) %>%   summarise(accuracy = mean(auc))
  mean_auc_generated$n <- mean_auc_generated$total_size
  
  acc_graphs <- list()
  auc_graphs <- list()
  for (classifier in unique(metric_generated$method)) {
    ss_est_generated <- curve_fit(acc_table = mean_acc_generated[mean_acc_generated$method == classifier,],  n_target = n_target, annotation = c("Accuracy", classifier))
    acc_graphs <- append(acc_graphs, list(ss_est_generated$graph))
    if(classifier != "KNN"){
      ss_est_generated <- curve_fit(acc_table = mean_auc_generated[mean_auc_generated$method == classifier,],  n_target = n_target, annotation = c("AUC", classifier))
      auc_graphs <- append(auc_graphs, list(ss_est_generated$graph))
    }
  }
  print(ggarrange(acc_graphs[[1]], 
                  acc_graphs[[2]], 
                  acc_graphs[[3]], 
                  acc_graphs[[4]], 
                  acc_graphs[[5]],
                  nrow = 3, ncol = 2, common.legend = T, legend = "bottom"))
}