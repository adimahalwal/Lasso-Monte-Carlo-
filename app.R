library(shiny)
library(tidyverse)
library(MASS)
library(glmnet)

# Define UI for the Shiny app
ui <- fluidPage(
  titlePanel("Monte Carlo Simulation: OLS vs LASSO (with CV)"),
  
  sidebarLayout(
    sidebarPanel(
      sliderInput("n_obs", "Number of Observations:", min = 50, max = 200, value = 80, step = 10),
      sliderInput("n_sim", "Number of Simulations:", value = 100, min = 50, max = 1000, step = 50),
      sliderInput("sigma_sq", "Error Variance (σ²):", min = 0, max = 20, value = 10, step = 1),
      selectInput("correlation", "Covariate Correlation (ρ):", 
                  choices = c("Low (0.1)" = 0.1, "High (0.7)" = 0.7), selected = 0.1),
      actionButton("run_sim", "Run Simulation", class = "btn-primary")
    ),
    
    mainPanel(
      h3("Simulation Results"),
      tableOutput("summary_table"),
      plotOutput("mse_plot"),
      plotOutput("bias_plot"),
      plotOutput("variance_plot"),
      plotOutput("bv_tradeoff_plot")
    )
  )
)

# Define server logic for the Shiny app
server <- function(input, output) {
  
  # Helper function to generate correlated predictors
  generate_X <- function(n, p, rho) {
    Sigma <- matrix(rho, nrow = p, ncol = p) + diag(1 - rho, p)
    mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  }
  
  # Reactive expression to run the simulation
  simulation_results <- eventReactive(input$run_sim, {
    n <- input$n_obs
    n_train <- floor(0.8 * n)
    n_test <- n - n_train
    p <- 25
    n_sim <- input$n_sim
    sigma_sq <- input$sigma_sq
    rho <- as.numeric(input$correlation)
    
    true_beta <- c(runif(5, 0.5, 1), rep(0, p - 5))
    
    mse_ols <- mse_lasso_cv <- numeric(n_sim)
    bias_ols <- bias_lasso_cv <- numeric(n_sim)
    variance_ols <- variance_lasso_cv <- numeric(n_sim)
    
    withProgress(message = "Running Simulations...", value = 0, {
      for (i in seq_len(n_sim)) {
        incProgress(1 / n_sim, detail = paste("Simulation", i, "of", n_sim))
        X <- generate_X(n, p, rho)
        X_train <- X[1:n_train, ]
        X_test <- X[(n_train + 1):n, ]
        epsilon_train <- rnorm(n_train, mean = 0, sd = sqrt(sigma_sq))
        epsilon_test <- rnorm(n_test, mean = 0, sd = sqrt(sigma_sq))
        Y_train <- X_train %*% true_beta + epsilon_train
        Y_test <- X_test %*% true_beta + epsilon_test
        
        ols_model <- lm(Y_train ~ ., data = as.data.frame(X_train))
        ols_predictions <- predict(ols_model, newdata = as.data.frame(X_test))
        mse_ols[i] <- mean((Y_test - ols_predictions)^2)
        bias_ols[i] <- mean(Y_test - ols_predictions)
        variance_ols[i] <- var(ols_predictions)
        
        lasso_model_cv <- cv.glmnet(X_train, Y_train, alpha = 1)
        lasso_predictions_cv <- predict(lasso_model_cv, newx = X_test, s = "lambda.min")
        mse_lasso_cv[i] <- mean((Y_test - lasso_predictions_cv)^2)
        bias_lasso_cv[i] <- mean(Y_test - lasso_predictions_cv)
        variance_lasso_cv[i] <- var(lasso_predictions_cv)
      }
    })
    
    list(
      mse_ols = mse_ols,
      mse_lasso_cv = mse_lasso_cv,
      bias_ols = bias_ols,
      bias_lasso_cv = bias_lasso_cv,
      variance_ols = variance_ols,
      variance_lasso_cv = variance_lasso_cv
    )
  })
  
  # Generate summary table
  output$summary_table <- renderTable({
    results <- simulation_results()
    data.frame(
      Metric = c("MSE", "Bias", "Variance"),
      OLS_Mean = c(mean(results$mse_ols), mean(results$bias_ols), mean(results$variance_ols)),
      OLS_SD = c(sd(results$mse_ols), sd(results$bias_ols), sd(results$variance_ols)),
      LASSO_Mean = c(mean(results$mse_lasso_cv), mean(results$bias_lasso_cv), mean(results$variance_lasso_cv)),
      LASSO_SD = c(sd(results$mse_lasso_cv), sd(results$bias_lasso_cv), sd(results$variance_lasso_cv))
    )
  }, rownames = FALSE)
  
  # Kernel Density Plots
  output$mse_plot <- renderPlot({
    results <- simulation_results()
    df <- data.frame(
      Model = rep(c("OLS", "LASSO (CV)"), each = length(results$mse_ols)),
      MSE = c(results$mse_ols, results$mse_lasso_cv)
    )
    ggplot(df, aes(x = MSE, fill = Model)) +
      geom_density(alpha = 0.5) +
      labs(title = "MSE Density", x = "MSE", y = "Density") +
      scale_fill_manual(values = c("red", "yellow"))
  })
  
  output$bias_plot <- renderPlot({
    results <- simulation_results()
    df <- data.frame(
      Model = rep(c("OLS", "LASSO (CV)"), each = length(results$bias_ols)),
      Bias = c(results$bias_ols, results$bias_lasso_cv)
    )
    ggplot(df, aes(x = Bias, fill = Model)) +
      geom_density(alpha = 0.5) +
      labs(title = "Bias Density", x = "Bias", y = "Density") +
      scale_fill_manual(values = c("red", "yellow"))
  })
  
  output$variance_plot <- renderPlot({
    results <- simulation_results()
    df <- data.frame(
      Model = rep(c("OLS", "LASSO (CV)"), each = length(results$variance_ols)),
      Variance = c(results$variance_ols, results$variance_lasso_cv)
    )
    ggplot(df, aes(x = Variance, fill = Model)) +
      geom_density(alpha = 0.5) +
      labs(title = "Variance Density", x = "Variance", y = "Density") +
      scale_fill_manual(values = c("red", "yellow"))
  })
  
  output$bv_tradeoff_plot <- renderPlot({
    results <- simulation_results()
    df <- data.frame(
      Metric = rep(c("MSE", "Bias", "Variance"), each = length(results$mse_ols) * 2),
      Model = rep(c("OLS", "LASSO (CV)"), times = 3 * length(results$mse_ols)),
      Value = c(results$mse_ols, results$mse_lasso_cv,
                results$bias_ols, results$bias_lasso_cv,
                results$variance_ols, results$variance_lasso_cv)
    )
    ggplot(df, aes(x = Value, fill = Model)) +
      geom_density(alpha = 0.5) +
      facet_wrap(~ Metric, scales = "free", ncol = 1) +
      labs(title = "Bias-Variance Tradeoff (Density)", x = "Value", y = "Density") +
      scale_fill_manual(values = c("red", "yellow"))
  })
}

# Run the application
shinyApp(ui = ui, server = server)

