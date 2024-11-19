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
      plotOutput("lambda_effect_plot")
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
    
    lambda_seq <- 10^seq(-4, 0, length.out = 100) # Lambda values for LASSO
    mse_lasso <- sum_beta <- dropped_covariates <- matrix(0, nrow = n_sim, ncol = length(lambda_seq))
    optimal_lambdas <- numeric(n_sim)
    
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
        
        # LASSO with cross-validation
        lasso_model_cv <- cv.glmnet(X_train, Y_train, alpha = 1, lambda = lambda_seq)
        optimal_lambdas[i] <- lasso_model_cv$lambda.min
        
        for (j in seq_along(lambda_seq)) {
          lasso_predictions <- predict(lasso_model_cv, newx = X_test, s = lambda_seq[j])
          mse_lasso[i, j] <- mean((Y_test - lasso_predictions)^2)
          
          # Calculate the sum of absolute beta coefficients
          beta_coeff <- coef(lasso_model_cv, s = lambda_seq[j])[-1]  # Exclude intercept
          sum_beta[i, j] <- sum(abs(beta_coeff))
          
          # Count the number of non-zero coefficients
          coef_count <- sum(beta_coeff != 0)
          dropped_covariates[i, j] <- p - coef_count
        }
      }
    })
    
    list(
      lambda_seq = lambda_seq,
      mse_lasso = mse_lasso,
      sum_beta = sum_beta,
      dropped_covariates = dropped_covariates,
      optimal_lambdas = optimal_lambdas
    )
  })
  
  # Generate summary table
  output$summary_table <- renderTable({
    results <- simulation_results()
    lambda_avg <- mean(results$optimal_lambdas)
    avg_dropped_covariates <- mean(results$dropped_covariates[, which.min(colMeans(results$mse_lasso))])
    avg_sum_beta <- mean(results$sum_beta[, which.min(colMeans(results$mse_lasso))])
    data.frame(
      Metric = c("Average Optimal Lambda", "Average Covariates Dropped", "Average Sum of Beta Coefficients"),
      Value = c(lambda_avg, avg_dropped_covariates, avg_sum_beta)
    )
  }, rownames = FALSE)
  
  # Plot the effect of lambda on MSE, Sum of Betas, and Dropped Covariates
  output$lambda_effect_plot <- renderPlot({
    results <- simulation_results()
    lambda_seq <- results$lambda_seq
    mse_mean <- colMeans(results$mse_lasso)
    sum_beta_mean <- colMeans(results$sum_beta)
    dropped_covariates_mean <- colMeans(results$dropped_covariates)
    optimal_lambda <- mean(results$optimal_lambdas)
    
    df <- data.frame(
      Lambda = lambda_seq,
      MSE = mse_mean,
      SumBeta = sum_beta_mean,
      DroppedCovariates = dropped_covariates_mean
    )
    
    ggplot(df, aes(x = Lambda)) +
      geom_line(aes(y = MSE, color = "MSE"), size = 1.2) +
      geom_line(aes(y = SumBeta, color = "Sum of Betas"), size = 1.2) +
      geom_line(aes(y = DroppedCovariates, color = "Covariates Dropped"), size = 1.2) +
      scale_y_continuous(
        name = "MSE / Sum of Betas",
        sec.axis = sec_axis(~ ., name = "Average Covariates Dropped")
      ) +
      scale_x_log10() +
      geom_vline(xintercept = optimal_lambda, linetype = "dashed", color = "red", size = 1) +
      labs(title = "Effect of Lambda on MSE, Sum of Betas, and Dropped Covariates",
           x = "Lambda (log scale)", color = "Metric") +
      theme_minimal()
  })
}

# Run the application
shinyApp(ui = ui, server = server)
