library(dplyr)
library(tidyr)
library(corrplot)
library(ggplot2)
library(lightgbm)
library(caret)
library(Metrics)

# Custom function to calculate the mode (most frequent value)
calculate_mode <- function(x) {
  x <- na.omit(x)  # Remove NAs for mode calculation
  unique_x <- unique(x)
  unique_x[which.max(tabulate(match(x, unique_x)))]
}

# Read the data
train_data <- read.csv("train_data.csv")
test_data <- read.csv("test_data.csv")

# Define variable types
continuous_vars <- c("ann_prm_amt", "newest_veh_age", "home_lot_sq_footage", 
                     "household_policy_counts", "X12m_call_history", "tenure_at_snapshot")
categorical_vars <- c("acq_method", "bi_limit_group", "channel", "digital_contact_ind", "geo_group", 
                      "has_prior_carrier", "household_group", "pay_type_code", "pol_edeliv_ind", 
                      "prdct_sbtyp_grp", "product_sbtyp", "telematics_ind", "trm_len_mo")
target_var <- "call_counts"


###########################################
# 1. Data Preprocessing
###########################################

# Function to prepare data for LightGBM
prepare_data_lgb <- function(data, continuous_vars, categorical_vars) {
  
  # 1. Convert categorical variables to factors
  data <- data %>%
    mutate(across(all_of(categorical_vars), factor))
  
  # 2. Handle missing values
  data <- data %>%
    mutate(
      newest_veh_age = ifelse(newest_veh_age == -20, NA, newest_veh_age),
      telematics_ind = ifelse(telematics_ind == -2, NA, telematics_ind),
      pol_edeliv_ind = ifelse(pol_edeliv_ind == -2, NA, pol_edeliv_ind)
    ) %>%
    # Impute missing values with median for numeric variables and mode for factors
    mutate(
      across(all_of(continuous_vars), ~replace_na(., median(., na.rm = TRUE))),
      across(all_of(categorical_vars), ~replace_na(
        ., if (is.numeric(.)) as.numeric(calculate_mode(.)) else as.character(calculate_mode(.))
      ))
    )
  
  # 3. Create dummy variables for categorical columns
  dummy_data <- model.matrix(~ . - 1, data = data[categorical_vars]) # -1 to remove intercept
  dummy_data <- as.data.frame(dummy_data) # Convert to data frame
  
  # 4. Combine with numeric columns
  final_data <- cbind(data[continuous_vars], dummy_data)
  
  return(final_data)
} 

processed_train <- prepare_data_lgb(train_data, continuous_vars, categorical_vars)
processed_test <- prepare_data_lgb(test_data, continuous_vars, categorical_vars)

# Split train-validation
set.seed(42)
train_indices <- createDataPartition(train_data[[target_var]], p = 0.8, list = FALSE)
dtrain <- lgb.Dataset(data = as.matrix(processed_train[train_indices, ]), label = train_data[[target_var]][train_indices], free_raw_data = FALSE)
dvalid <- lgb.Dataset(data = as.matrix(processed_train[-train_indices, ]), label = train_data[[target_var]][-train_indices], free_raw_data = FALSE)

###########################################
# 2. Hyperparameter Optimization
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
# 3. Model Training
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
# 4. Model Evaluation
###########################################

evaluate_model <- function(model, X, y = NULL) {
  # Check for best iteration
  num_iter <- if (!is.null(model$best_iter)) model$best_iter else model$current_iter()
  
  # Make predictions
  preds <- predict(model, as.matrix(X), num_iteration = num_iter)
  
  if (!is.null(y)) {
    # Ensure dimensions match
    if (length(y) != length(preds)) stop("Mismatch between `y` and `preds` lengths.")
    
    # Calculate metrics
    metrics <- tryCatch({
      calculate_metrics(y, preds)
    }, error = function(e) {
      warning("Error in metric calculation: ", e$message)
      list()
    })
    
    # Feature importance
    importance <- tryCatch({
      lgb.importance(model, percentage = TRUE)
    }, error = function(e) {
      warning("Error calculating feature importance: ", e$message)
      NULL
    })
    
    # Feature importance plot
    importance_plot <- if (!is.null(importance)) {
      lgb.plot.importance(importance, top_n = 10, measure = "Gain")
    } else {
      NULL
    }
    
    # Training history plot
    history_plot <- tryCatch({
      plot_training_history(model)
    }, error = function(e) {
      warning("Error plotting training history: ", e$message)
      NULL
    })
    
    # Actual vs Predicted plot
    pred_plot <- ggplot(data.frame(Actual = y, Predicted = preds), 
                        aes(x = Actual, y = Predicted)) +
      geom_point(alpha = 0.5) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
      labs(title = "Actual vs Predicted Values", 
           x = "Actual", 
           y = "Predicted") +
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
    # Return only predictions
    return(list(predictions = preds))
  }
}

# Custom evaluation metrics
calculate_metrics <- function(y_true, y_pred) {
  list(
    rmse = tryCatch(sqrt(mean((y_true - y_pred)^2)), error = function(e) NA),
    mae = tryCatch(mean(abs(y_true - y_pred)), error = function(e) NA),
    poisson_ll = tryCatch(mean(y_true * log(y_pred) - y_pred), error = function(e) NA),
    r2 = tryCatch(cor(y_true, y_pred)^2, error = function(e) NA)
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


# Assuming `model` is your trained LightGBM model
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
print(train_results$metrics)

# Evaluate model performance on the validation set
validation_results <- evaluate_model(model, processed_train[-train_indices, ], train_data[[target_var]][-train_indices])

# Validation Metrics
print(validation_results$metrics)


# Evaluate model on test data
test_results <- evaluate_model(model, processed_test)  
print("Test Predictions (first 10):")
print(head(test_results$predictions, 10))

# Convert test predictions to a data frame
test_predictions_df <- data.frame(Predictions = test_results$predictions)

# Convert test predictions to a data frame with 'id' and 'predict' columns
submission <- data.frame(
  id = seq_len(nrow(test_data)),                 
  Predict = test_predictions_df$Predictions 
)

# Print first few rows of the submission file
print("Sample Submission Format (first 10 rows):")
print(head(submission, 10))

# Optionally, save the submission to a CSV file
write.csv(submission, "submissionn.csv", row.names = FALSE)


# Save the trained model
save_model(model, "lightgbm_model.rds")

#Try Neural Network(Fit Forward or back prop)
#Use Gini Impurity as mode cost
#Use output from your base learner as your predictor
#Use HPO for GBM, XGBOOST,NN (Genetic Algorithm). Searches a combinatorial space

#Projection to lower dimension (PCA, Partial Least Square, Project pursuit, ppr package in R)

#E-M algorithm with missing imputation
#Try MissForest non-parametric implementation for mixed-type data
#Missing X in test set: use K-nearest Neighbor

#Stacking to determine which model to choose

#Keep track of the F-Measure for the classifier







