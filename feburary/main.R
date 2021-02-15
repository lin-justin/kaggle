library(here)
library(tidymodels)
library(tidyverse)
library(ggthemes)
library(gridExtra)
library(vip)
library(doParallel)
library(glmnet)
library(rules)
library(bestNormalize)

all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

set.seed(123)

# ================ EXPLORATORY DATA ANALYSIS ================

# Read in the data

original_train_data <- read_csv("data/train.csv")

original_test_data <- read_csv("data/test.csv")

# Remove the id column

train_data <- original_train_data %>%
  select(-id)

test_data <- original_test_data %>%
  select(-id)

# Check for missing values

is.na(train_data) %>%
  colSums()

is.na(test_data) %>%
  colSums()

# Extract character/categorical variables
# Plot barplots to see if imbalanced or not

categorical_variables <- train_data %>%
  select_if(is.character)

plot_categorical_data <- function(data, column) {
  ggplot(data, aes_string(x = column)) + 
    geom_bar(fill = "cornflowerblue")
}

cat_plots <- lapply(colnames(categorical_variables), 
                    plot_categorical_data, 
                    data = categorical_variables)

cat_g <- grid.arrange(grobs = cat_plots, nrow = 5)

# Extract numerical variables
# Plot density plots to see distribution, check for skewness

numerical_variables <- train_data %>%
  select_if(is.numeric)

plot_numerica_data <- function(data, column) {
  ggplot(data, aes_string(x = column)) + 
    geom_density(fill = "cornflowerblue")
}

numeric_plots <- lapply(colnames(numerical_variables),
                        plot_numerica_data,
                        data = numerical_variables)

num_g <- grid.arrange(grobs = numeric_plots, nrow = 5)

# If a plots folder/directory does not exist, create one
if (!dir.exists("plots/")) {dir.create("plots/")}

ggsave("plots/categorical-plots.png", cat_g)
ggsave("plots/numeric-plots.png", num_g)

# ================ END OF EDA ================

# ================ MODELING ================

# Reference: http://www.rebeccabarter.com/blog/2020-03-25_machine_learning/#tune-the-parameters

set.seed(123)
data_split <- initial_split(train_data, prop = 0.80)

# Preprocessing recipe

set.seed(123)
preprocessing_rec <- recipe(target ~., data = training(data_split)) %>%
  step_bestNormalize(all_numeric(), -target) %>%
  step_scale(all_numeric(), -target) %>%
  step_other(all_nominal(), threshold = 10000) %>%
  step_integer(all_nominal()) %>%
  step_corr(all_predictors()) %>%
  step_lincomb(all_predictors()) %>%
  step_nzv(all_predictors())

# See what the preprocessed data looks like

data_train_preprocessed <- preprocessing_rec %>%
  prep(training(data_split)) %>%
  juice()

glimpse(data_train_preprocessed)

# Cross-validation folds
set.seed(123)
data_cv_folds <- vfold_cv(data = training(data_split), v = 5)

# Initialize Regularized Regression model

lm_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Hyperparameters for tuning

lm_params <- dials::parameters(
  penalty(),
  mixture()
)

set.seed(123)
lm_grid <- dials::grid_random(
  lm_params,
  size = 3
)

# Set up workflow

lm_wf <- workflow() %>%
  add_recipe(preprocessing_rec) %>%
  add_model(lm_model)

# Tune hyperparameters

lm_tuned <- lm_wf %>%
  tune_grid(
    resamples = data_cv_folds,
    grid = lm_grid,
    metrics = metric_set(rmse, rsq, mae),
    control = control_grid(verbose = TRUE)
)

# Select best hyperparameter

param_best <- lm_tuned %>%
  select_best(metric = "rmse")

# Finalize workflow 

lm_wf <- lm_wf %>%
  finalize_workflow(param_best)

# Evaluate model on test data from split

lm_fit <- lm_wf %>%
  last_fit(data_split)

test_performance <- lm_fit %>%
  collect_metrics()
test_performance

# Re-fit on training data and use final/best model
# to predict on unseen test data

final_model <- fit(lm_wf, train_data)

predictions <- predict(final_model, new_data = test_data)
glimpse(predictions)

submission <- cbind(original_test_data, predictions)
glimpse(submission)

submission <- rename(submission, target = .pred)
glimpse(submission)

submission <- submission %>%
  select(id, target)

glimpse(submission)

write_csv(submission, "lm_submission.csv")
