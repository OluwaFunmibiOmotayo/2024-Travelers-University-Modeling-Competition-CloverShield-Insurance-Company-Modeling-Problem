

###########################################
# 0. Introduction
###########################################

# This script implements a LightGBM model for predicting `call_counts` based on customer data.
# It includes data preprocessing, feature engineering, hyperparameter tuning, model training, 
# and evaluation. The script is enhanced to document each stage of the process, addressing key 
# questions outlined in the presentation instructions.

###########################################
# Libraries and Setup
###########################################

library(dplyr)
library(tidyr)
library(corrplot)
library(ggplot2)
library(lightgbm)
library(caret)
library(Metrics)

###########################################
# Custom function to calculate the mode
###########################################

calculate_mode <- function(x) {
  x <- na.omit(x)  # Remove NAs for mode calculation
  unique_x <- unique(x)
  unique_x[which.max(tabulate(match(x, unique_x)))]
}

###########################################
# 1. Data Loading and Overview
###########################################

train_data <- read.csv("train_data.csv")
test_data <- read.csv("test_data.csv")

# Explore the structure of the data
summary(train_data)
summary(test_data)

###########################################
# 2. Data Preprocessing
###########################################

# Define variable types
continuous_vars <- c("ann_prm_amt", "newest_veh_age", "home_lot_sq_footage", 
                     "household_policy_counts", "X12m_call_history", "tenure_at_snapshot")
categorical_vars <- c("acq_method", "bi_limit_group", "channel", "digital_contact_ind", "geo_group", 
                      "has_prior_carrier", "household_group", "pay_type_code", "pol_edeliv_ind", 
                      "prdct_sbtyp_grp", "product_sbtyp", "telematics_ind", "trm_len_mo")
target_var <- "call_counts"


# Function to prepare data for LightGBM
prepare_data_lgb <- function(data, continuous_vars, categorical_vars) {
  
  # 1. Convert categorical variables to factors
  data <- data %>%
    mutate(across(all_of(categorical_vars), factor))
  
  # 2. Handle missing values and special values for specific variables
  data <- data %>%
    # Replace -20 in newest_veh_age, -2 in telematics_ind and pol_edeliv_ind with NA for imputation
    mutate(
      newest_veh_age = ifelse(newest_veh_age == -20, NA, newest_veh_age),
      telematics_ind = ifelse(telematics_ind == -2, NA, telematics_ind),
      pol_edeliv_ind = ifelse(pol_edeliv_ind == -2, NA, pol_edeliv_ind)
    ) %>%
    # Create indicator variables for missing or special categories
    mutate(
      is_newest_veh_age_missing = ifelse(is.na(newest_veh_age), 1, 0),
      is_telematics_ind_missing = ifelse(is.na(telematics_ind), 1, 0),
      is_pol_edeliv_ind_missing = ifelse(is.na(pol_edeliv_ind), 1, 0)
    ) %>%
    # Impute missing values for continuous and categorical variables
    mutate(
      across(all_of(continuous_vars), ~replace_na(., median(., na.rm = TRUE))),
      across(all_of(categorical_vars), ~replace_na(
        ., if (is.numeric(.)) as.numeric(calculate_mode(.)) else as.character(calculate_mode(.))
      ))
    )
  
  # 3. Create dummy variables for categorical columns
  dummy_data <- model.matrix(~ . - 1, data = data[categorical_vars]) # -1 to remove intercept
  dummy_data <- as.data.frame(dummy_data) # Convert to data frame
  
  # 4. Combine with numeric columns and indicator variables, with clear column names
  final_data <- cbind(
    data[continuous_vars],
    dummy_data,
    is_newest_veh_age_missing = data[["is_newest_veh_age_missing"]],
    is_telematics_ind_missing = data[["is_telematics_ind_missing"]],
    is_pol_edeliv_ind_missing = data[["is_pol_edeliv_ind_missing"]]
  )
  
  return(final_data)
} 

# Calculate % missing
missing_percentage <- sapply(train_data, function(x) mean(is.na(x)) * 100)
missing_percentage

# Prepare training and validation data
processed_train <- prepare_data_lgb(train_data, continuous_vars, categorical_vars)
processed_test <- prepare_data_lgb(test_data, continuous_vars, categorical_vars)

# Calculate % zeros for response variable (call_counts)
zero_percentage_call_counts <- mean(processed_train$call_counts == 0) * 100
zero_percentage_call_counts

###########################################
# 3. Feature Selection
###########################################
#Correlation between Continuous and Target Variable
continuous_associations <- sapply(continuous_vars, function(var) cor(processed_train[[var]], train_data$call_counts, use = "complete.obs"))
continuous_associations

# Correlation analysis between the continuous variables
corr_matrix <- cor(processed_train[continuous_vars])
corrplot(corr_matrix, method = "circle", type = "upper", tl.cex = 0.7)

# Exclude the continuous variables to get categorical variables in the new data
cat_vars <- setdiff(names(processed_train), continuous_vars)

# Calculate Cramér's V for each categorical variable
library(vcd)  # Ensure the 'vcd' package is loaded
categorical_associations <- sapply(cat_vars, function(var) {
  # Create a contingency table
  table_var <- table(processed_train[[var]], train_data$call_counts)
  
  # Compute Cramér's V
  cramer_v <- assocstats(table_var)$cramer
  return(cramer_v)
})

# Output the results
categorical_associations

###########################################
# 4. Model Development
###########################################

# Split train-validation
set.seed(42)
train_indices <- createDataPartition(train_data[[target_var]], p = 0.8, list = FALSE)
dtrain <- lgb.Dataset(data = as.matrix(processed_train[train_indices, ]), label = train_data[[target_var]][train_indices], free_raw_data = FALSE)
dvalid <- lgb.Dataset(data = as.matrix(processed_train[-train_indices, ]), label = train_data[[target_var]][-train_indices], free_raw_data = FALSE)


###########################################
# 4.1 Hyperparameter Optimization
###########################################

tune_hyperparameters <- function(dtrain, dvalid, nfolds = 5) {
  # Grid of parameters to try
  param_grid <- expand.grid(
    num_leaves = c(31, 63, 127),
    learning_rate = c(0.01, 0.05, 0.1),
    feature_fraction = c(0.7, 0.8, 0.9),
    min_data_in_leaf = c(20, 50, 100),
    max_depth = c(-1, 5, 10)
  )
  
  best_score <- Inf
  best_params <- NULL
  
  for(i in 1:nrow(param_grid)) {
    current_params <- list(
      objective = "poisson",
      metric = "poisson",
      boosting = "gbdt",
      num_leaves = param_grid$num_leaves[i],
      learning_rate = param_grid$learning_rate[i],
      feature_fraction = param_grid$feature_fraction[i],
      min_data_in_leaf = param_grid$min_data_in_leaf[i],
      max_depth = param_grid$max_depth[i]
    )
    
    tryCatch({
      cv_results <- lgb.cv(
        params = current_params,
        data = dtrain,
        nfold = nfolds,
        nrounds = 1000,
        early_stopping_rounds = 50,
        stratified = TRUE,
        eval_freq = 50,
        verbose = 0
      )
      best_score_cv <- cv_results$best_score
      
      if(best_score_cv < best_score) {
        best_score <- best_score_cv
        best_params <- current_params
        cat(sprintf("New best score: %.4f with parameters:\n", best_score))
        print(best_params)
        cat("\n")
      }
    }, error = function(e) {
      cat(sprintf("Error with parameters at row %d: %s\n", i, e$message))
    })
  }
  
  if (is.null(best_params)) {
    stop("No valid parameters found during tuning")
  }
  
  return(best_params)
}

###########################################
# 4.2 Model Training
###########################################

# Train Model
train_model <- function(dtrain, dvalid, params, seed = 42) {
  set.seed(seed)
  model <- lgb.train(
    params = params,
    data = dtrain,
    nrounds = 1000,
    valids = list(training = dtrain, valid = dvalid),
    early_stopping_rounds = 50,
    eval_freq = 10,
    verbose = 1
  )
  return(model)
}

###########################################
# 4.3 Model Evaluation
###########################################

evaluate_model <- function(model, X, y = NULL) {
  # Make predictions
  preds <- predict(model, as.matrix(X), num_iteration = model$best_iter)
  
  if (!is.null(y)) {
    # Calculate metrics
    metrics <- calculate_metrics(y, preds)
    
    # Feature importance
    importance <- lgb.importance(model, percentage = TRUE)
    importance_plot <- lgb.plot.importance(importance, top_n = 10, measure = "Gain")
    
    # Training history plot
    history_plot <- plot_training_history(model)
    
    # Actual vs Predicted plot
    pred_plot <- ggplot(data.frame(Actual = y, Predicted = preds), 
                        aes(x = Actual, y = Predicted)) +
      geom_point(alpha = 0.5) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
      labs(title = "Actual vs Predicted Values", 
           x = "Actual Call Counts", 
           y = "Predicted Call Counts") +
      theme_minimal()
    
    return(list(
      metrics = metrics,
      importance = importance,
      plots = list(
        importance = importance_plot,
        history = history_plot,
        predictions = pred_plot
      )
    ))
  } else {
    # Return only predictions for test data
    return(list(
      predictions = preds
    ))
  }
}

# Custom evaluation metrics
calculate_metrics <- function(y_true, y_pred) {
  list(
    rmse = rmse(y_true, y_pred),
    mae = mae(y_true, y_pred),
    poisson_ll = mean(y_true * log(y_pred) - y_pred),
    r2 = cor(y_true, y_pred)^2
  )
}

# Custom plotting function for training history
plot_training_history <- function(model) {
  data <- data.frame(
    Iteration = 1:length(model$record_evals$valid$poisson$eval),
    Training = model$record_evals$training$poisson$eval,
    Validation = model$record_evals$valid$poisson$eval
  )
  
  ggplot(data, aes(x = Iteration)) +
    geom_line(aes(y = Training, color = "Training")) +
    geom_line(aes(y = Validation, color = "Validation")) +
    labs(title = "Training History",
         y = "Poisson Loss",
         color = "Dataset") +
    theme_minimal()
}

###########################################
# 5. Main Execution
###########################################

# Tune hyperparameters
best_params <- tune_hyperparameters(dtrain, dvalid)

# Train model with best parameters
model <- train_model(dtrain, dvalid, best_params)


best_params_value <- model$params

# Access specific parameters
cat("Best Parameters:\n")
cat("num_leaves:", best_params_value$num_leaves, "\n")
cat("learning_rate:", best_params_value$learning_rate, "\n")
cat("feature_fraction:", best_params_value$feature_fraction, "\n")
cat("min_data_in_leaf:", best_params_value$min_data_in_leaf, "\n")
cat("max_depth:", best_params_value$max_depth, "\n")

# Evaluate model on training data
train_results <- evaluate_model(model, processed_train[train_indices, ], train_data[[target_var]][train_indices])
print("Training Metrics:")
print(train_results$metrics)


# Evaluate model on test data
test_results <- evaluate_model(model, processed_test)  
print("Test Predictions (first 10):")
print(head(test_results$predictions, 10))

# Convert test predictions to a data frame
test_predictions_df <- data.frame(Predictions = test_results$predictions)

# Convert test predictions to a data frame with 'id' and 'predict' columns
submission2 <- data.frame(
  id = seq_len(nrow(test_data)),                 
  Predict = test_predictions_df$Predictions 
)

# Print first few rows of the submission file
print("Sample Submission Format (first 10 rows):")
print(head(submission, 10))

# Optionally, save the submission to a CSV file
write.csv(submission2, "submission2new.csv", row.names = FALSE)


# Save the trained model
save_model(model, "lightgbm_model.rds")



