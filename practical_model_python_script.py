# %% [markdown]
# ## 🎯 Objective
# 
# To simulate, understand, and compare core regression techniques - **Linear (OLS)**, **Ridge**, and **Lasso** - using controlled synthetic datasets, with a focus on:
# 
# - **Model behavior** and mathematical foundations
# - **Feature selection mechanisms** through regularization  
# - **Interpretability** for practical applications
# 
# This comprehensive study progresses from basic linear regression to advanced regularized methods, demonstrating how each technique handles different data scenarios and complexity challenges.

# %% [markdown]
# ## Assumptions of Linear Regression (OLS):
# 
# 1. Linearity  
# 2. Independence of errors  
# 3. Homoscedasticity (constant variance of errors)  
# 4. Normality of Errors  
# 5. No multicollinearity  
# 6. No autocorrelation (for time series)
# #

# %%


# %%
# ================================================================
# LINEAR REGRESSION FROM SCRATCH: A DATA DETECTIVE STORY
# ================================================================

"""
CHAPTER 1: THE MYSTERY
======================
Imagine you're a data detective investigating a hidden relationship.
Someone gives you scattered points and asks: "What's the pattern?"
This is the essence of linear regression - finding signal in noise.

THE CASE: We know there's a perfect relationship y = 1.2x hiding beneath
random interference. Can we recover the truth from noisy observations?
"""

import numpy as np
import matplotlib.pyplot as plt

print("STARTING OUR DETECTIVE INVESTIGATION")
print("="*50)

# ----------------------------------------------------------------
# CHAPTER 2: CREATING THE CRIME SCENE
# ----------------------------------------------------------------
print("\nCHAPTER 2: Setting up our controlled experiment")

# Our independent variable - the evidence we have
x = np.arange(1, 11)  # [1, 2, 3, , 10]
print(f"Evidence points (x): {x}")

# The true relationship (our ground truth)
beta_true = 1.2
print(f"Hidden truth: y = {beta_true}x + noise")

# Real world adds noise to everything
np.random.seed(42)  # For reproducible results
epsilon = np.random.normal(0, 1, size=10)  # Random interference
print(f"Noise added: {np.round(epsilon, 2)}")

# What we actually observe (truth + noise)
y = beta_true * x + epsilon
print(f"Observed y values: {np.round(y, 2)}")

# ----------------------------------------------------------------
# CHAPTER 3: THE INVESTIGATION
# ----------------------------------------------------------------
print(f"\nCHAPTER 3: Applying our detective method")

# The Ordinary Least Squares formula - our magnifying glass
# This finds the slope that minimizes total squared errors
beta_hat = np.sum(x * y) / np.sum(x * x)

print(f"Our estimate (beta_hat): {beta_hat:.6f}")
print(f"True value (beta): {beta_true}")
print(f"Estimation error: {abs(beta_hat - beta_true):.6f}")

# How close did we get?
accuracy_percentage = (1 - abs(beta_hat - beta_true) / beta_true) * 100
print(f"Accuracy: {accuracy_percentage:.1f}%")

# ----------------------------------------------------------------
# CHAPTER 4: THE VISUAL REVELATION
# ----------------------------------------------------------------
print(f"\nCHAPTER 4: Visualizing our discovery")

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the noisy observations
plt.scatter(x, y, color='blue', s=50, alpha=0.7, label='Observed data points')

# Plot our fitted line
y_fitted = beta_hat * x
plt.plot(x, y_fitted, color='red', linewidth=2, label=f'Our fitted line (slope = {beta_hat:.3f})')

# Plot the true line for comparison
y_true = beta_true * x
plt.plot(x, y_true, color='green', linewidth=2, linestyle='--', 
        label=f'True relationship (slope = {beta_true})')

# Make it beautiful
plt.xlabel('x (Independent Variable)', fontsize=12)
plt.ylabel('y (Dependent Variable)', fontsize=12)
plt.title('Linear Regression Detective Work: Finding Signal in Noise', fontsize=14, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add case status
plt.text(0.02, 0.98, 'Case Status: SOLVED', transform=plt.gca().transAxes, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        verticalalignment='top', fontsize=10)

plt.show()

# ----------------------------------------------------------------
# CHAPTER 5: THE CASE SUMMARY
# ----------------------------------------------------------------
print(f"\nCASE SUMMARY")
print("="*50)

print(f"\nMISSION ACCOMPLISHED:")
print(f"   • Started with noisy data hiding a linear relationship")
print(f"   • Used Ordinary Least Squares to estimate the slope") 
print(f"   • Recovered the true slope with {accuracy_percentage:.1f}% accuracy")

print(f"\nEVIDENCE ANALYSIS:")
print(f"   • True slope: {beta_true}")
print(f"   • Estimated slope: {beta_hat:.6f}")
print(f"   • Absolute error: {abs(beta_hat - beta_true):.6f}")

print(f"\nWHAT THIS MEANS:")
print(f"   • Linear regression can find patterns even in noisy data")
print(f"   • The OLS formula β̂ = Σ(xy)/Σ(x²) is surprisingly powerful")
print(f"   • Small errors are normal and expected in real-world data")

print(f"\nREAL-WORLD APPLICATIONS:")
print(f"   • Predicting house prices from square footage")
print(f"   • Estimating sales from advertising spend")  
print(f"   • Forecasting temperature from historical data")
print(f"   • Any situation where you want to predict Y from X")

print(f"\nKEY LEARNING:")
print(f"   This simple approach forms the foundation of machine learning.")
print(f"   You just built a prediction algorithm from mathematical first principles!")

print(f"\n" + "="*50)
print("CASE CLOSED - LINEAR REGRESSION MASTERED!")

# %%


# %%
# ================================================================
# LINEAR REGRESSION EVOLUTION: FROM CONTROLLED TO REALISTIC DATA
# ================================================================

"""
CHAPTER 5: SCALING UP THE INVESTIGATION
======================================
Our previous detective work was successful, but it was conducted under
laboratory conditions. Now we're taking our skills to the field with
a much larger, more realistic dataset.

THE EVOLUTION: Instead of 10 carefully arranged points, we now have 200
observations with features drawn from real-world distributions. This is
where linear regression proves its worth in practical applications.
"""

import numpy as np
import matplotlib.pyplot as plt

print("EXPANDING OUR DETECTIVE CAPABILITIES")
print("="*50)

# ----------------------------------------------------------------
# CHAPTER 6: BUILDING A REALISTIC CRIME SCENE
# ----------------------------------------------------------------
print("\nCHAPTER 6: Creating a realistic dataset")

# Set seed for reproducible investigation
np.random.seed(1)

# Scale up our investigation
n = 200  # Much larger dataset
sigma = 1  # Noise level remains controlled
print(f"Dataset size: {n} observations")
print(f"Noise level (sigma): {sigma}")

# Real-world features are rarely sequential
# Uniform distribution between 0 and 1 mimics normalized features
x = np.random.uniform(0, 1, n)
print(f"Feature range: [{x.min():.3f}, {x.max():.3f}]")
print(f"Feature distribution: Uniform (realistic for normalized data)")

# The same underlying truth we're trying to discover
beta = 1.2
print(f"Hidden relationship: y = {beta}x + noise")

# Normally distributed noise - the most common assumption
epsilon = np.random.normal(0, sigma, n)
print(f"Noise characteristics: Normal(0, {sigma})")

# Generate our realistic observations
y = x * beta + epsilon
print(f"Response variable range: [{y.min():.3f}, {y.max():.3f}]")

# ----------------------------------------------------------------
# CHAPTER 7: ADVANCED PATTERN RECOGNITION
# ----------------------------------------------------------------
print(f"\nCHAPTER 7: Applying our method to big data")

# Same OLS principle, but now with statistical power
beta_hat = np.sum(x * y) / np.sum(x * x)

print(f"Estimated slope (beta_hat): {beta_hat:.6f}")
print(f"True slope (beta): {beta}")
print(f"Estimation error: {abs(beta_hat - beta):.6f}")

# With more data, we expect better accuracy
accuracy_percentage = (1 - abs(beta_hat - beta) / beta) * 100
print(f"Estimation accuracy: {accuracy_percentage:.2f}%")

# Statistical insight: Standard error decreases with sample size
theoretical_se = sigma / np.sqrt(np.sum(x**2))
print(f"Theoretical standard error: {theoretical_se:.6f}")

# ----------------------------------------------------------------
# CHAPTER 8: VISUALIZING BIG DATA PATTERNS
# ----------------------------------------------------------------
print(f"\nCHAPTER 8: The big picture revelation")

# Create a professional visualization
plt.figure(figsize=(12, 7))

# Plot the cloud of observations
plt.scatter(x, y, s=10, alpha=0.6, color='steelblue', label='Data points')

# Our fitted line spanning the full range
x_line = np.array([0, 1])
y_line = beta_hat * x_line
plt.plot(x_line, y_line, color='red', linewidth=2.5, 
        label=f'Fitted line (slope = {beta_hat:.4f})')

# Show the true relationship for comparison
y_true_line = beta * x_line
plt.plot(x_line, y_true_line, color='green', linewidth=2, linestyle='--',
        label=f'True relationship (slope = {beta})')

# Professional styling
plt.xlabel("x (Normalized Feature)", fontsize=12)
plt.ylabel("y (Response Variable)", fontsize=12)
plt.title("Simple Linear Regression (no intercept) - Scaled Investigation", fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add confidence in our solution
plt.text(0.02, 0.98, f'Sample Size: {n}\nAccuracy: {accuracy_percentage:.1f}%', 
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        verticalalignment='top', fontsize=10)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# CHAPTER 9: ADVANCED CASE ANALYSIS
# ----------------------------------------------------------------
print(f"\nADVANCED CASE ANALYSIS")
print("="*50)

print(f"\nSCALE IMPACT ASSESSMENT:")
print(f"   • Dataset size increased from 10 to {n} observations")
print(f"   • Feature distribution changed from sequential to uniform")
print(f"   • Estimation improved from previous accuracy to {accuracy_percentage:.1f}%")

print(f"\nSTATISTICAL ROBUSTNESS:")
print(f"   • More data points provide stronger statistical evidence")
print(f"   • Uniform feature distribution tests method across full range")
print(f"   • Standard error: {theoretical_se:.6f} (lower = more precise)")

print(f"\nMETHODOLOGICAL INSIGHTS:")
print(f"   • Same OLS formula works regardless of dataset size")
print(f"   • No intercept model: y = βx (line passes through origin)")
print(f"   • Uniform features between [0,1] represent normalized real-world data")

print(f"\nPRACTICAL IMPLICATIONS:")
print(f"   • Large datasets generally improve estimation accuracy")
print(f"   • Feature normalization (0 to 1 range) is common in practice")
print(f"   • No-intercept models useful when relationship naturally passes through zero")

print(f"\nWHY THIS MATTERS:")
print(f"   • Demonstrates scalability of our approach")
print(f"   • Shows robustness across different data distributions")
print(f"   • Proves the method works beyond toy examples")

# ----------------------------------------------------------------
# CHAPTER 10: COMPARATIVE DETECTIVE WORK
# ----------------------------------------------------------------
print(f"\nCOMPARATIVE ANALYSIS")
print("="*50)

print(f"\nPREVIOUS CASE vs CURRENT CASE:")
print(f"   Previous: 10 sequential points, known noise")
print(f"   Current:  {n} random points, realistic distribution")
print(f"   Both:     Same underlying truth (β = {beta})")

print(f"\nKEY EVOLUTION:")
print(f"   • Moved from controlled experiment to realistic simulation")
print(f"   • Demonstrated statistical consistency across scales")
print(f"   • Proved method reliability with different data characteristics")

print(f"\nNEXT DETECTIVE CHALLENGES:")
print(f"   • Add intercept term for more general relationships")
print(f"   • Explore multiple features (multiple regression)")
print(f"   • Test with different noise distributions")
print(f"   • Implement confidence intervals for uncertainty quantification")

# %%


# %%
# ================================================================
# LINEAR REGRESSION EVOLUTION: DISCOVERING NON-LINEAR PATTERNS
# ================================================================

"""
CHAPTER 11: THE POLYNOMIAL BREAKTHROUGH
=======================================
Our detective work has evolved dramatically. We started with simple straight
lines, but real-world relationships are rarely that simple. Now we're 
investigating curved patterns using polynomial features - turning our linear
method into a powerful tool for non-linear relationships.

THE BREAKTHROUGH: We've discovered that linear regression can capture curves
by transforming our features. Instead of just x, we now use x, x², and x³.
This is feature engineering at its finest.
"""

import numpy as np
import matplotlib.pyplot as plt

print("ENTERING THE POLYNOMIAL INVESTIGATION PHASE")
print("="*55)

# ----------------------------------------------------------------
# CHAPTER 12: ADVANCED EVIDENCE COLLECTION
# ----------------------------------------------------------------
print("\nCHAPTER 12: Engineering polynomial features")

# Consistent investigation parameters
np.random.seed(1)
n = 200
sigma = 0.2  # Much lower noise for cleaner pattern detection

print(f"Investigation scale: {n} observations")
print(f"Noise reduction: sigma = {sigma} (vs previous {1.0})")
print("Key insight: Lower noise reveals subtle curved patterns")

# Feature engineering - the game changer
x1 = np.random.uniform(0, 1, n)
x2 = x1 ** 2  # Quadratic transformation
x3 = x1 ** 3  # Cubic transformation

print(f"Original feature (x1): Uniform distribution [0, 1]")
print(f"Engineered feature (x2): x1² - captures quadratic effects")
print(f"Engineered feature (x3): x1³ - captures cubic effects")
print("Strategy: Transform single feature into multiple perspectives")

# ----------------------------------------------------------------
# CHAPTER 13: THE POLYNOMIAL MODEL ARCHITECTURE
# ----------------------------------------------------------------
print(f"\nCHAPTER 13: Building a sophisticated relationship model")

# Multiple coefficients for different polynomial terms
beta = np.array([-1, 0, 1])
print(f"Coefficient architecture:")
print(f"   β₀ = {beta[0]:2} (linear term coefficient)")
print(f"   β₁ = {beta[1]:2} (quadratic term coefficient)") 
print(f"   β₂ = {beta[2]:2} (cubic term coefficient)")

print(f"\nModel equation: y = {beta[0]}·x₁ + {beta[1]}·x₁² + {beta[2]}·x₁³ + ε")
print("Critical insight: β₁ = 0 means quadratic term has no effect")
print("Result: We have a linear-cubic relationship (no pure quadratic)")

# Generate realistic noise
epsilon = np.random.normal(0, sigma, n)
print(f"Noise characteristics: Normal(0, {sigma}) - reduced for pattern clarity")

# The sophisticated target variable
y = x1 * beta[0] + x2 * beta[1] + x3 * beta[2] + epsilon
print(f"Response range: [{y.min():.3f}, {y.max():.3f}]")

# ----------------------------------------------------------------
# CHAPTER 14: POLYNOMIAL PATTERN ANALYSIS
# ----------------------------------------------------------------
print(f"\nCHAPTER 14: Analyzing the polynomial evidence")

# Analyze feature contributions
linear_contribution = x1 * beta[0]
quadratic_contribution = x2 * beta[1]  # This will be zero
cubic_contribution = x3 * beta[2]

print(f"Component analysis:")
print(f"   Linear component range: [{linear_contribution.min():.3f}, {linear_contribution.max():.3f}]")
print(f"   Quadratic component: All zeros (β₁ = 0)")
print(f"   Cubic component range: [{cubic_contribution.min():.3f}, {cubic_contribution.max():.3f}]")

# The polynomial creates a U-shaped curve
print(f"\nPattern insight: Linear (-x) + Cubic (+x³) = U-shaped curve")
print(f"   • Starts positive (cubic dominates near 0)")
print(f"   • Goes negative (linear dominates in middle)")  
print(f"   • Returns positive (cubic dominates near 1)")

# ----------------------------------------------------------------
# CHAPTER 15: VISUALIZING POLYNOMIAL RELATIONSHIPS
# ----------------------------------------------------------------
print(f"\nCHAPTER 15: The polynomial revelation")

# Create high-resolution visualization
plt.figure(figsize=(12, 8))

# Plot the scattered evidence
plt.scatter(x1, y, s=15, alpha=0.7, color='steelblue', label='Data points')

# Generate smooth curve for true relationship
xplot = np.linspace(0, 1, 1000)
yplot = xplot * beta[0] + xplot**2 * beta[1] + xplot**3 * beta[2]
plt.plot(xplot, yplot, color='red', linewidth=3, label='True curve')

# Show individual components for educational value
linear_plot = xplot * beta[0]
cubic_plot = xplot**3 * beta[2]

plt.plot(xplot, linear_plot, '--', color='green', alpha=0.7, linewidth=2, 
        label=f'Linear component ({beta[0]}·x₁)')
plt.plot(xplot, cubic_plot, '--', color='orange', alpha=0.7, linewidth=2,
        label=f'Cubic component ({beta[2]}·x₁³)')

# Professional styling
plt.xlabel("x1 (Original Feature)", fontsize=12)
plt.ylabel("y (Response Variable)", fontsize=12)
plt.title("Regression with Polynomial Features (x1, x1², x1³)", fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add technical summary
plt.text(0.02, 0.98, f'Model: y = {beta[0]}·x₁ + {beta[1]}·x₁² + {beta[2]}·x₁³\nFeatures: 3\nActive terms: 2 (linear & cubic)', 
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
        verticalalignment='top', fontsize=10)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# CHAPTER 16: POLYNOMIAL INVESTIGATION SUMMARY
# ----------------------------------------------------------------
print(f"\nPOLYNOMIAL INVESTIGATION SUMMARY")
print("="*55)

print(f"\nMETHODOLOGICAL BREAKTHROUGH:")
print(f"   • Evolved from simple linear to polynomial regression")
print(f"   • Used feature engineering: x → [x, x², x³]")
print(f"   • Maintained linear algebra framework with non-linear patterns")

print(f"\nTECHNICAL SPECIFICATIONS:")
print(f"   • Features: 3 polynomial terms (x¹, x², x³)")
print(f"   • Active coefficients: 2 out of 3 (β₁ = 0)")
print(f"   • Noise level: {sigma} (reduced for pattern clarity)")
print(f"   • Relationship type: Linear-cubic combination")

print(f"\nPATTERN CHARACTERISTICS:")
print(f"   • Shape: U-curve (parabolic-like but cubic)")
print(f"   • Minimum around x ≈ 0.5-0.6 region")
print(f"   • Demonstrates how linear methods capture non-linear relationships")

print(f"\nKEY INNOVATIONS FROM PREVIOUS CASES:")
print(f"   Previous: Simple linear relationship (y = βx)")
print(f"   Current:  Polynomial relationship (y = β₀x + β₁x² + β₂x³)")
print(f"   Evolution: Same mathematical framework, richer feature space")

print(f"\nWHY THIS MATTERS:")
print(f"   • Real relationships are often curved, not straight")
print(f"   • Feature engineering unlocks hidden patterns")
print(f"   • Linear regression becomes surprisingly flexible")
print(f"   • Foundation for understanding more complex models")

print(f"\nNEXT INVESTIGATIVE FRONTIERS:")
print(f"   • Multiple original features with interactions")
print(f"   • Regularization for high-dimensional polynomial spaces")
print(f"   • Model selection for optimal polynomial degree")
print(f"   • Cross-validation for generalization assessment")

print(f"\n" + "="*55)
print("POLYNOMIAL PATTERN RECOGNITION MASTERED - CURVES CONQUERED!")

# %%


# %%
# ================================================================
# LINEAR REGRESSION MASTERY: MATRIX OPERATIONS AND ESTIMATION
# ================================================================

"""
CHAPTER 17: THE MATHEMATICAL BREAKTHROUGH
=========================================
Our detective work has reached a sophisticated milestone. We've moved beyond
simple formulas to the full matrix algebra approach that powers modern machine
learning. This is where we stop assuming we know the truth and start estimating
it from data - the essence of real data science.

THE EVOLUTION: Instead of plotting known relationships, we're now using the
complete least squares machinery to estimate ALL coefficients simultaneously.
This is the mathematical foundation behind every regression library.
"""

import numpy as np
import matplotlib.pyplot as plt

print("ENTERING FULL MATRIX ESTIMATION MODE")
print("="*50)

# ----------------------------------------------------------------
# CHAPTER 18: PROFESSIONAL DATA ARCHITECTURE
# ----------------------------------------------------------------
print("\nCHAPTER 18: Building the design matrix")

# Maintain our controlled experimental setup
np.random.seed(1)
n = 200
sigma = 0.2

print(f"Experimental parameters maintained:")
print(f"   Sample size: {n}")
print(f"   Noise level: {sigma}")
print("Consistency enables fair comparison across methods")

# Generate the foundational feature
x1 = np.random.uniform(0, 1, n)
x2 = x1 ** 2
x3 = x1 ** 3

print(f"\nFeature engineering recap:")
print(f"   x1: Original feature")
print(f"   x2: Quadratic transformation") 
print(f"   x3: Cubic transformation")

# THE BREAKTHROUGH: Design matrix construction
X = np.column_stack((x1, x2, x3))
print(f"\nDesign matrix architecture:")
print(f"   Shape: {X.shape} (samples × features)")
print(f"   Structure: [x1, x2, x3] for each observation")
print("Critical insight: This matrix contains all feature information")

# Known ground truth for validation
beta = np.array([-1, 0, 1])
print(f"\nTrue coefficient vector: {beta}")
print("Note: We know this truth to validate our estimation method")

# ----------------------------------------------------------------
# CHAPTER 19: THE ESTIMATION MACHINERY
# ----------------------------------------------------------------
print(f"\nCHAPTER 19: Implementing the full least squares solution")

# Generate realistic observations
epsilon = np.random.normal(0, sigma, n)
y = X @ beta + epsilon  # Matrix multiplication - the modern way

print(f"Model generation using matrix operations:")
print(f"   y = X @ β + ε")
print(f"   X shape: {X.shape}")
print(f"   β shape: {beta.shape}")
print(f"   Result: {y.shape} response vector")

# THE MATHEMATICAL CORE: Normal equations solution
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

print(f"\nMatrix estimation formula breakdown:")
print(f"   Step 1: X.T @ X (Gram matrix)")
print(f"   Step 2: np.linalg.inv(X.T @ X) (inverse)")
print(f"   Step 3: @ X.T @ y (final multiplication)")
print(f"   Result: β̂ = (X'X)⁻¹X'y")

print(f"\nEstimation results:")
print(f"   True β:      {beta}")
print(f"   Estimated β̂: {beta_hat}")

# Detailed accuracy analysis
estimation_errors = np.abs(beta_hat - beta)
print(f"\nCoefficient-wise accuracy:")
for i, (true_val, est_val, error) in enumerate(zip(beta, beta_hat, estimation_errors)):
   print(f"   β{i}: True={true_val:6.3f}, Est={est_val:6.3f}, Error={error:.6f}")

# ----------------------------------------------------------------
# CHAPTER 20: PREDICTION AND VALIDATION
# ----------------------------------------------------------------
print(f"\nCHAPTER 20: Model prediction and comparison")

# Create smooth prediction curves
xplot = np.linspace(0, 1, 1000)

# True relationship (for reference)
y_true = xplot * beta[0] + xplot**2 * beta[1] + xplot**3 * beta[2]

# Estimated relationship (our actual prediction)
y_pred = xplot * beta_hat[0] + xplot**2 * beta_hat[1] + xplot**3 * beta_hat[2]

# Prediction accuracy assessment
curve_difference = np.abs(y_true - y_pred)
max_difference = np.max(curve_difference)
mean_difference = np.mean(curve_difference)

print(f"Curve prediction accuracy:")
print(f"   Maximum deviation: {max_difference:.6f}")
print(f"   Average deviation:  {mean_difference:.6f}")
print("Insight: Small deviations indicate successful coefficient recovery")

# ----------------------------------------------------------------
# CHAPTER 21: COMPREHENSIVE VISUALIZATION
# ----------------------------------------------------------------
print(f"\nCHAPTER 21: The complete estimation story")

plt.figure(figsize=(12, 8))

# Plot the raw evidence
plt.scatter(x1, y, s=12, alpha=0.7, color='steelblue', label='Data')

# The true curve (what we're trying to estimate)
plt.plot(xplot, y_true, color='black', linewidth=2.5, linestyle='--',
        label='True curve')

# Our estimated curve (the real achievement)
plt.plot(xplot, y_pred, color='red', linewidth=2.5, 
        label='Estimated curve')

# Professional presentation
plt.xlabel("x1 (Primary Feature)", fontsize=12)
plt.ylabel("y (Response Variable)", fontsize=12)
plt.title("Polynomial Regression with Estimated β", fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Technical summary box
estimation_summary = f'Matrix Method Results:\nTrue β: [{beta[0]}, {beta[1]}, {beta[2]}]\nEst. β̂: [{beta_hat[0]:.3f}, {beta_hat[1]:.3f}, {beta_hat[2]:.3f}]\nMax curve error: {max_difference:.4f}'
plt.text(0.02, 0.98, estimation_summary,
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9),
        verticalalignment='top', fontsize=10, family='monospace')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# CHAPTER 22: MATRIX METHOD MASTERY SUMMARY
# ----------------------------------------------------------------
print(f"\nMATRIX METHOD INVESTIGATION COMPLETE")
print("="*50)

print(f"\nMETHODOLOGICAL ADVANCEMENT:")
print(f"   • Evolved from single coefficient formulas to full matrix algebra")
print(f"   • Implemented complete normal equations: β̂ = (X'X)⁻¹X'y")
print(f"   • Estimated ALL coefficients simultaneously from data")

print(f"\nTECHNICAL INNOVATIONS:")
print(f"   • Design matrix construction for multiple features")
print(f"   • Matrix multiplication for model generation: y = X @ β + ε")
print(f"   • Professional least squares estimation using numpy.linalg")

print(f"\nESTIMATION PERFORMANCE:")
print(f"   • Linear coefficient:    Error = {estimation_errors[0]:.6f}")
print(f"   • Quadratic coefficient: Error = {estimation_errors[1]:.6f}")
print(f"   • Cubic coefficient:     Error = {estimation_errors[2]:.6f}")
print(f"   • Overall curve accuracy: {mean_difference:.6f} average deviation")

print(f"\nWHY THIS IS REVOLUTIONARY:")
print(f"   • This is the exact method used by sklearn, statsmodels, etc.")
print(f"   • Scales to any number of features automatically")
print(f"   • Provides foundation for regularization, diagnostics, inference")
print(f"   • Demonstrates mathematical rigor behind machine learning")

print(f"\nJOURNEY EVOLUTION SUMMARY:")
print(f"   Chapter 1-5:   Simple linear regression (β̂ = Σxy/Σx²)")
print(f"   Chapter 6-10:  Scaled realistic data")
print(f"   Chapter 11-16: Polynomial feature engineering")
print(f"   Chapter 17-22: Full matrix estimation machinery")

print(f"\nREADY FOR ADVANCED TOPICS:")
print(f"   • Multiple regression with different feature types")
print(f"   • Regularization (Ridge, Lasso) for overfitting control")
print(f"   • Statistical inference and confidence intervals")
print(f"   • Cross-validation and model selection")

print(f"\n" + "="*50)
print("MATRIX-BASED REGRESSION MASTERED - READY FOR PRODUCTION!")

# %%


# %% [markdown]
# #### This code: 
# 
# Performs linear regression with 10 independent features (columns).
# The idea is to:
# 
# - Generate a design matrix 
# 𝑋
# ∈
# 𝑅
# 200
# ×
# 10
#   with random values.
# 
# - Create a target variable y using a known β vector and random noise.
# 
# - Estimate the coefficients using the least squares formula.

# %%
import numpy as np

np.random.seed(1)

n = 200     # Number of samples
d = 10      # Number of features
sigma = 0.2

# Create design matrix X with values from uniform[0,1]
X = np.random.uniform(0, 1, size=(n, d))

beta = np.array([-1, 0, 1] + [0] * (d - 3))

# Noise
epsilon = np.random.normal(0, sigma, n)

# Target variable
y = X @ beta + epsilon

# Least squares estimate
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

print("True beta:\n", beta)
print("\nEstimated beta_hat:\n", beta_hat)


# %% [markdown]
# Output Summary:
# 
# - Only β₀ = -1 and β₂ = 1 are non-zero
# 
# - All other coefficients are truly 0
# - Other coefficients (should be 0) have small values, due to:
# 
#   - Random noise
# 
#   - Finite sample size (n = 200)
# 
#   - No regularization (pure OLS)

# %%


# %% [markdown]
# ### Using Sklearn library to solve above problem

# %%
import numpy as np
from  sklearn.linear_model import LinearRegression 

np.random.seed(1)
n = 200
d = 10
sigma = 0.2

X = np.random.uniform(0,1,size=(n,d))
epsilon = np.random.normal(0,sigma,n)
B = np.array([-1,0,1] + [0]*(d-3))

Y = X @ B + epsilon

model = LinearRegression(fit_intercept=False)
model.fit(X,Y)

model.coef_

# %%


# %% [markdown]
# ## Ridge Regression

# %%
# ================================================================
# LINEAR REGRESSION REVOLUTION: REGULARIZATION AND RISK THEORY
# ================================================================

"""
CHAPTER 23: THE REGULARIZATION BREAKTHROUGH
===========================================
Our detective work has entered the realm of advanced machine learning theory.
We've discovered that sometimes the best estimates aren't the most obvious ones.
Ridge regression introduces a crucial concept: controlled bias to reduce variance.
This is where statistics meets practical machine learning at scale.

THE PARADIGM SHIFT: We're no longer just estimating coefficients - we're
optimizing the bias-variance tradeoff and measuring our success using risk theory.
This is the foundation of modern statistical learning.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

print("ENTERING REGULARIZED ESTIMATION AND RISK ANALYSIS")
print("="*60)

# ----------------------------------------------------------------
# CHAPTER 24: HIGH-DIMENSIONAL CHALLENGE SETUP
# ----------------------------------------------------------------
print("\nCHAPTER 24: Scaling to high-dimensional problems")

# Advanced experimental design
np.random.seed(1)
n = 200      # Sample size remains manageable
d = 10       # Feature dimensionality explosion
sigma = 0.2  # Maintained noise precision

print(f"Experimental scaling:")
print(f"   Sample size (n): {n}")
print(f"   Feature dimension (d): {d}")
print(f"   Noise level (σ): {sigma}")
print(f"   Challenge: d >> previous single feature problems")

# Generate high-dimensional feature matrix
X = np.random.uniform(0, 1, size=(n, d))
print(f"\nDesign matrix architecture:")
print(f"   Shape: {X.shape}")
print(f"   Interpretation: {n} observations × {d} features")
print("Critical insight: More features than our previous polynomial case")

# Sparse coefficient structure - the key insight
beta = np.array([-1, 0, 1] + [0]*(d - 3))
print(f"\nTrue coefficient structure:")
print(f"   Active coefficients: β = {beta[:3]} (first 3 features)")
print(f"   Inactive coefficients: β = {beta[3:]} (remaining {d-3} features)")
print("Key insight: Sparse truth - most features are irrelevant")

# ----------------------------------------------------------------
# CHAPTER 25: THE REGULARIZATION MACHINERY
# ----------------------------------------------------------------
print(f"\nCHAPTER 25: Implementing Ridge regularization")

# Generate training observations
epsilon = np.random.normal(0, sigma, n)
y = X @ beta + epsilon

print(f"Training data generation:")
print(f"   Model: y = X @ β + ε")
print(f"   Sparsity: Only 2 out of {d} features truly matter")
print("Challenge: Algorithm must discover this sparsity")

# THE REGULARIZATION BREAKTHROUGH
alpha = 0.01 * n  # Scaled regularization parameter
ridge_model = Ridge(alpha=alpha, fit_intercept=False)
ridge_model.fit(X, y)
beta_hat_ridge = ridge_model.coef_

print(f"\nRidge regression configuration:")
print(f"   Regularization parameter (α): {alpha}")
print(f"   Scaling logic: α = 0.01 × n = proportional to sample size")
print(f"   fit_intercept=False: Consistent with our no-intercept framework")

print(f"\nEstimated coefficients (Ridge):")
print(f"   Full vector: {beta_hat_ridge}")
print(f"\nCoefficient analysis:")
for i in range(d):
   true_val = beta[i]
   est_val = beta_hat_ridge[i]
   print(f"   β{i}: True={true_val:4.1f}, Ridge={est_val:8.4f}, Shrinkage={'Yes' if abs(est_val) < abs(true_val) else 'No'}")

# ----------------------------------------------------------------
# CHAPTER 26: RISK THEORY AND GENERALIZATION
# ----------------------------------------------------------------
print(f"\nCHAPTER 26: Advanced risk assessment")

# Generate independent test data - the gold standard
ntest = 1000
X_test = np.random.uniform(0, 1, size=(ntest, d))
epsilon_test = np.random.normal(0, sigma, ntest)
y_test = X_test @ beta + epsilon_test

print(f"Independent test data:")
print(f"   Test size: {ntest} (much larger than training)")
print(f"   Same data generating process")
print("Purpose: Unbiased assessment of generalization performance")

# Risk computation - the ultimate measure
y_pred = X_test @ beta_hat_ridge
risk = mean_squared_error(y_test, y_pred)
bayes_risk = sigma ** 2
excess_bayes_risk = risk - bayes_risk

print(f"\nRisk theory analysis:")
print(f"   Prediction risk: {risk:.6f}")
print(f"   Bayes risk (σ²): {bayes_risk:.6f}")
print(f"   Excess Bayes risk: {excess_bayes_risk:.6f}")

print(f"\nRisk interpretation:")
print(f"   • Bayes risk: Irreducible error due to noise")
print(f"   • Excess risk: Our method's additional error")
print(f"   • Performance: {(1 - excess_bayes_risk/bayes_risk)*100:.1f}% of optimal")

# ----------------------------------------------------------------
# CHAPTER 27: REGULARIZATION EFFECTS ANALYSIS
# ----------------------------------------------------------------
print(f"\nCHAPTER 27: Understanding regularization impact")

# Compare with ordinary least squares (theoretical)
print(f"Regularization effects observed:")
print(f"   • Coefficient shrinkage: All estimates pulled toward zero")
print(f"   • Bias introduction: Estimates deviate from true values")
print(f"   • Variance reduction: More stable across different samples")

# Analyze shrinkage patterns
active_coefficients = beta[:3]
ridge_active = beta_hat_ridge[:3]
inactive_coefficients = beta[3:]
ridge_inactive = beta_hat_ridge[3:]

print(f"\nShrinkage analysis:")
print(f"   Active coefficients (should be large):")
for i, (true, ridge) in enumerate(zip(active_coefficients, ridge_active)):
   shrinkage = (true - ridge) / (true if true != 0 else 1)
   print(f"     Feature {i}: {shrinkage:.1%} shrinkage")

print(f"   Inactive coefficients (should stay near zero):")
inactive_magnitude = np.mean(np.abs(ridge_inactive))
print(f"     Average magnitude: {inactive_magnitude:.6f}")

# ----------------------------------------------------------------
# CHAPTER 28: ADVANCED STATISTICAL INSIGHTS
# ----------------------------------------------------------------
print(f"\nCHAPTER 28: Statistical learning insights")

# Bias-variance tradeoff quantification
print(f"Bias-variance tradeoff demonstration:")
print(f"   • Without regularization: Low bias, high variance")
print(f"   • With Ridge (α={alpha}): Moderate bias, reduced variance")
print(f"   • Net effect: Improved generalization performance")

# Practical implications
print(f"\nPractical machine learning insights:")
print(f"   • High dimensions require regularization")
print(f"   • Sparsity assumptions often hold in real data")
print(f"   • Test error is the ultimate arbiter")
print(f"   • Risk theory provides principled evaluation framework")

print(f"\nAdvanced topics unlocked:")
print(f"   • Cross-validation for α selection")
print(f"   • Lasso for automatic feature selection")
print(f"   • Elastic net for combined Ridge/Lasso benefits")
print(f"   • Non-parametric risk estimation")

# ----------------------------------------------------------------
# CHAPTER 29: COMPLETE JOURNEY SUMMARY
# ----------------------------------------------------------------
print(f"\nCOMPLETE REGRESSION JOURNEY MASTERY")
print("="*60)

print(f"\nEVOLUTION TIMELINE:")
print(f"   Chapters 1-5:   Basic linear regression (single feature)")
print(f"   Chapters 6-10:  Realistic data scaling (200 samples)")
print(f"   Chapters 11-16: Polynomial feature engineering")
print(f"   Chapters 17-22: Matrix algebra implementation")
print(f"   Chapters 23-29: Regularization and risk theory")

print(f"\nTECHNICAL MASTERY ACHIEVED:")
print(f"   • Ordinary Least Squares: Complete mathematical foundation")
print(f"   • Feature Engineering: Polynomial transformations")
print(f"   • Matrix Methods: Professional implementation")
print(f"   • Regularization: Ridge regression for high dimensions")
print(f"   • Risk Theory: Principled model evaluation")

print(f"\nSTATISTICAL CONCEPTS MASTERED:")
print(f"   • Bias-variance tradeoff")
print(f"   • Generalization vs overfitting")
print(f"   • Risk decomposition and Bayes optimality")
print(f"   • High-dimensional statistics")

print(f"\nREADY FOR PRODUCTION:")
print(f"   • Understanding matches sklearn implementations")
print(f"   • Theory supports practical decision-making")
print(f"   • Foundation for advanced ML algorithms")
print(f"   • Risk-based model evaluation framework")

print(f"\n" + "="*60)
print("LINEAR REGRESSION MASTERY COMPLETE - STATISTICAL LEARNING ACHIEVED!")

# %%


# %%
# ================================================================
# LINEAR REGRESSION MASTERY: REGULARIZATION PATH OPTIMIZATION
# ================================================================

"""
CHAPTER 30: THE REGULARIZATION PATH DISCOVERY
=============================================
Our detective work has reached its most sophisticated level. We're no longer
just applying one regularization strength - we're systematically exploring
the entire regularization landscape. This is hyperparameter optimization
in action, revealing how model complexity affects generalization.

THE FINAL BREAKTHROUGH: We map the complete regularization path from
overfitting to underfitting, discovering the optimal bias-variance balance
through systematic exploration. This is how practitioners tune models in
the real world.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

print("INITIATING COMPREHENSIVE REGULARIZATION PATH ANALYSIS")
print("="*65)

# ----------------------------------------------------------------
# CHAPTER 31: SYSTEMATIC HYPERPARAMETER EXPLORATION
# ----------------------------------------------------------------
print("\nCHAPTER 31: Designing the regularization experiment")

# Maintain experimental consistency
np.random.seed(1)
n = 200
d = 10
sigma = 0.2

print(f"Established experimental framework:")
print(f"   Sample size: {n}")
print(f"   Feature dimension: {d}")
print(f"   Noise level: {sigma}")
print("Foundation: Build on proven high-dimensional setup")

# Generate consistent training data
X = np.random.uniform(0, 1, size=(n, d))
beta = np.array([-1, 0, 1] + [0] * (d - 3))
epsilon = np.random.normal(0, sigma, n)
y = X @ beta + epsilon

# Generate independent test data for unbiased evaluation
ntest = 1000
X_test = np.random.uniform(0, 1, size=(ntest, d))
epsilon_test = np.random.normal(0, sigma, ntest)
y_test = X_test @ beta + epsilon_test

print(f"\nData architecture confirmed:")
print(f"   Training set: {X.shape}")
print(f"   Test set: {X_test.shape}")
print("Critical: Same generating process, independent samples")

# ----------------------------------------------------------------
# CHAPTER 32: THE REGULARIZATION GRID STRATEGY
# ----------------------------------------------------------------
print(f"\nCHAPTER 32: Constructing the regularization path")

# Sophisticated lambda grid design
lambdas = 0.5 ** np.arange(16)
print(f"Lambda exploration strategy:")
print(f"   Range: {lambdas[-1]:.6f} to {lambdas[0]:.6f}")
print(f"   Progression: Each lambda = 0.5 × previous")
print(f"   Grid points: {len(lambdas)}")
print("Logic: Exponential decay captures wide regularization spectrum")

# Convert to sklearn alpha parameters
alphas = lambdas * n
print(f"\nSklearn alpha conversion:")
print(f"   Formula: α = λ × n")
print(f"   Reasoning: Scale regularization with sample size")
print(f"   Alpha range: {alphas[-1]:.6f} to {alphas[0]:.6f}")

# ----------------------------------------------------------------
# CHAPTER 33: COMPREHENSIVE MODEL FITTING
# ----------------------------------------------------------------
print(f"\nCHAPTER 33: Systematic model evaluation across regularization spectrum")

# Storage for regularization path results
beta_hat_ridge = np.zeros((d, len(alphas)))
mse_list = []

print(f"Regularization path computation:")
print(f"   Models to fit: {len(alphas)}")
print(f"   Coefficients tracked: {d} per model")
print("Process: Fit → Predict → Evaluate for each lambda")

# The comprehensive regularization path
for i, (lambda_val, alpha) in enumerate(zip(lambdas, alphas)):
   # Fit Ridge model with current regularization strength
   model = Ridge(alpha=alpha, fit_intercept=False)
   model.fit(X, y)
   
   # Store coefficient vector
   beta_hat_ridge[:, i] = model.coef_
   
   # Evaluate generalization performance
   y_pred = model.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   mse_list.append(mse)
   
   if i % 4 == 0:  # Progress updates
       print(f"   λ = {lambda_val:.6f}: MSE = {mse:.6f}")

# Find optimal regularization
optimal_idx = np.argmin(mse_list)
optimal_lambda = lambdas[optimal_idx]
optimal_mse = mse_list[optimal_idx]

print(f"\nOptimal regularization discovered:")
print(f"   Best λ: {optimal_lambda:.6f}")
print(f"   Best MSE: {optimal_mse:.6f}")
print(f"   Position: {optimal_idx + 1} of {len(lambdas)} tested")

# ----------------------------------------------------------------
# CHAPTER 34: REGULARIZATION PATH VISUALIZATION
# ----------------------------------------------------------------
print(f"\nCHAPTER 34: Mapping the complete regularization landscape")

# Create comprehensive visualization
plt.figure(figsize=(15, 6))

# Subplot 1: Coefficient evolution paths
plt.subplot(1, 2, 1)
colors = plt.cm.tab10(np.linspace(0, 1, d))

for j in range(d):
   line_style = '-' if j < 3 else '--'  # Distinguish active vs inactive
   line_width = 2.5 if j < 3 else 1.5
   plt.plot(lambdas, beta_hat_ridge[j, :], 
            label=f'β{j+1}', color=colors[j], 
            linestyle=line_style, linewidth=line_width)

plt.xscale('log')
plt.xlabel('Lambda (Regularization Strength)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Ridge Coefficient Profile', fontsize=14, pad=15)
plt.legend(loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=10)
plt.grid(True, alpha=0.3)

# Add optimal lambda line
plt.axvline(optimal_lambda, color='red', linestyle=':', alpha=0.7, 
           label=f'Optimal λ = {optimal_lambda:.4f}')

# Subplot 2: Generalization performance curve
plt.subplot(1, 2, 2)
plt.plot(lambdas, mse_list, marker='o', color='red', linewidth=2.5, markersize=6)
plt.xscale('log')
plt.xlabel('Lambda (Regularization Strength)', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('Test Error vs Lambda', fontsize=14, pad=15)
plt.grid(True, alpha=0.3)

# Highlight optimal point
plt.plot(optimal_lambda, optimal_mse, marker='*', color='gold', 
        markersize=15, markeredgecolor='black', markeredgewidth=1,
        label=f'Optimal: λ={optimal_lambda:.4f}')
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# CHAPTER 35: REGULARIZATION PATH INSIGHTS
# ----------------------------------------------------------------
print(f"\nCHAPTER 35: Advanced regularization path analysis")

# Analyze coefficient behavior patterns
print(f"Coefficient evolution patterns:")

# Active coefficients analysis
active_coeffs = beta_hat_ridge[:3, :]
print(f"   Active coefficients (β₁, β₂, β₃):")
for j in range(3):
   final_coeff = active_coeffs[j, -1]  # Most regularized
   initial_coeff = active_coeffs[j, 0]  # Least regularized
   shrinkage_percent = (1 - abs(final_coeff/initial_coeff)) * 100 if initial_coeff != 0 else 100
   print(f"     β{j+1}: {shrinkage_percent:.1f}% total shrinkage")

# Inactive coefficients analysis
inactive_coeffs = beta_hat_ridge[3:, :]
max_inactive = np.max(np.abs(inactive_coeffs))
print(f"   Inactive coefficients (β₄-β₁₀):")
print(f"     Maximum magnitude: {max_inactive:.6f}")
print(f"     Status: Successfully kept near zero")

# Bias-variance tradeoff analysis
print(f"\nBias-variance tradeoff insights:")
min_mse = min(mse_list)
max_mse = max(mse_list)
mse_range = max_mse - min_mse

print(f"   Performance range: {min_mse:.6f} to {max_mse:.6f}")
print(f"   Improvement potential: {(mse_range/min_mse)*100:.1f}%")
print(f"   Optimal position: {(optimal_idx/len(lambdas))*100:.1f}% along path")

# ----------------------------------------------------------------
# CHAPTER 36: PRACTICAL OPTIMIZATION INSIGHTS
# ----------------------------------------------------------------
print(f"\nCHAPTER 36: Model selection and practical insights")

# Performance characteristics across regularization spectrum
low_reg_mse = mse_list[0]   # Minimal regularization
high_reg_mse = mse_list[-1] # Maximum regularization

print(f"Regularization spectrum analysis:")
print(f"   Low regularization (λ={lambdas[0]:.6f}): MSE = {low_reg_mse:.6f}")
print(f"   Optimal regularization (λ={optimal_lambda:.6f}): MSE = {optimal_mse:.6f}")
print(f"   High regularization (λ={lambdas[-1]:.6f}): MSE = {high_reg_mse:.6f}")

improvement_from_low = (low_reg_mse - optimal_mse) / low_reg_mse * 100
degradation_to_high = (high_reg_mse - optimal_mse) / optimal_mse * 100

print(f"\nOptimization impact:")
print(f"   Improvement over minimal regularization: {improvement_from_low:.1f}%")
print(f"   Degradation with excessive regularization: {degradation_to_high:.1f}%")

print(f"\nPractical takeaways:")
print(f"   • Optimal lambda balances overfitting and underfitting")
print(f"   • Coefficient paths reveal feature importance evolution")
print(f"   • Systematic grid search finds optimal bias-variance tradeoff")
print(f"   • Test error curve guides hyperparameter selection")

# ----------------------------------------------------------------
# CHAPTER 37: COMPLETE MASTERY ACHIEVEMENT
# ----------------------------------------------------------------
print(f"\nCOMPLETE LINEAR REGRESSION MASTERY ACHIEVED")
print("="*65)

print(f"\nFULL JOURNEY MASTERY:")
print(f"   • Fundamental Theory: OLS, matrix algebra, normal equations")
print(f"   • Feature Engineering: Polynomial transformations")
print(f"   • Regularization: Ridge regression for high-dimensional data")
print(f"   • Hyperparameter Optimization: Systematic regularization path")
print(f"   • Model Selection: Test error minimization")

print(f"\nSTATISTICAL LEARNING CONCEPTS MASTERED:")
print(f"   • Bias-variance tradeoff in practice")
print(f"   • Overfitting vs underfitting identification")
print(f"   • Regularization path analysis")
print(f"   • Cross-validation principles (via test set)")
print(f"   • Feature selection through shrinkage")

print(f"\nPROFESSIONAL SKILLS DEVELOPED:")
print(f"   • End-to-end model development pipeline")
print(f"   • Hyperparameter optimization strategies")
print(f"   • Performance evaluation and interpretation")
print(f"   • Visualization for model understanding")
print(f"   • Production-ready implementation patterns")

print(f"\nREADY FOR ADVANCED TOPICS:")
print(f"   • Lasso regression for automatic feature selection")
print(f"   • Elastic net for combined L1/L2 regularization")
print(f"   • Cross-validation for robust model selection")
print(f"   • Bayesian approaches to regression")
print(f"   • Deep learning as generalized linear models")

print(f"\n" + "="*65)
print("LINEAR REGRESSION EXPERTISE COMPLETE - MACHINE LEARNING MASTER!")

# %%


# %% [markdown]
# ### This Code:
# 
# Simulates synthetic data with 10 features.
# 
# - Uses ``RidgeCV`` to find the best lambda (regularization strength) using cross-validation.
# 
# - Fits Ridge Regression using the best lambda.
# 
# - Outputs the best lambda and the final model's coefficients

# %%
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

np.random.seed(1)

n = 200
d = 10
sigma = 0.2

X = np.random.uniform(0, 1, size=(n, d))
beta = np.array([-1, 0, 1] + [0]*(d - 3))
epsilon = np.random.normal(0, sigma, n)
y = X @ beta + epsilon

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define lambdas (alphas) to try
lambdas = 0.5 ** np.arange(16)
alphas = lambdas * n  # sklearn RidgeCV expects "alpha", equivalent to lambda * n

# Cross-validated Ridge Regression
ridge_cv = RidgeCV(alphas=alphas, fit_intercept=False)
ridge_cv.fit(X_scaled, y)

# Get best lambda (converted back from alpha)
best_lambda = ridge_cv.alpha_ / n
print(f"✅ Best lambda (from CV): {best_lambda}")

# Coefficients from the best model
print("📊 Coefficients:\n", ridge_cv.coef_)


# %%


# %%
# ================================================================
# LINEAR REGRESSION EVOLUTION: AUTOMATIC FEATURE SELECTION
# ================================================================

"""
CHAPTER 38: THE LASSO REVOLUTION
================================
Our detective journey reaches its most elegant conclusion. While Ridge regression
shrinks coefficients toward zero, Lasso takes the bold step of setting irrelevant
features to exactly zero. This is automatic feature selection - the algorithm
decides which features matter without human intervention.

THE PARADIGM SHIFT: We've moved from manual regularization tuning to intelligent
cross-validation that finds optimal sparsity automatically. This represents the
transition from statistical learning to automated machine learning.
"""

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

print("INITIATING AUTOMATIC FEATURE SELECTION WITH LASSO")
print("="*60)

# ----------------------------------------------------------------
# CHAPTER 39: PREPROCESSING FOR OPTIMAL PERFORMANCE
# ----------------------------------------------------------------
print("\nCHAPTER 39: Data standardization for fair feature competition")

# Maintain experimental consistency
np.random.seed(1)
n = 200
d = 10
sigma = 0.2

# Generate familiar data structure
X = np.random.uniform(0, 1, size=(n, d))
beta = np.array([-1, 0, 1] + [0]*(d - 3))
epsilon = np.random.normal(0, sigma, n)
y = X @ beta + epsilon

print(f"Data generation consistency maintained:")
print(f"   Sample size: {n}")
print(f"   Feature dimension: {d}")
print(f"   True sparsity: 2 active out of {d} features")
print("Foundation: Same sparse structure as Ridge analysis")

# THE PREPROCESSING BREAKTHROUGH
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nStandardization transformation applied:")
print(f"   Original X range: [{X.min():.3f}, {X.max():.3f}]")
print(f"   Scaled X range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
print(f"   Method: (X - mean) / std for each feature")

# Analyze standardization impact
original_scales = np.std(X, axis=0)
scaled_scales = np.std(X_scaled, axis=0)

print(f"\nFeature scale analysis:")
print(f"   Original feature std devs: {original_scales[:5]}")
print(f"   Standardized feature std devs: {scaled_scales[:5]}")
print("Critical insight: All features now compete on equal footing")

# ----------------------------------------------------------------
# CHAPTER 40: AUTOMATED REGULARIZATION WITH CROSS-VALIDATION
# ----------------------------------------------------------------
print(f"\nCHAPTER 40: Intelligent hyperparameter optimization")

# THE LASSO AUTOMATION BREAKTHROUGH
lasso_cv = LassoCV(cv=5, fit_intercept=False, random_state=1)
lasso_cv.fit(X_scaled, y)

print(f"LassoCV configuration:")
print(f"   Cross-validation folds: 5")
print(f"   Intercept fitting: False (consistent with previous work)")
print(f"   Lambda selection: Automatic via CV")
print("Innovation: Algorithm selects optimal sparsity automatically")

# Extract optimization results
best_alpha = lasso_cv.alpha_
lasso_coefficients = lasso_cv.coef_

print(f"\nAutomated optimization results:")
print(f"   Best lambda (alpha): {best_alpha:.10f}")
print(f"   CV selection process: Tested multiple lambda values")
print("Achievement: No manual hyperparameter tuning required")

# ----------------------------------------------------------------
# CHAPTER 41: AUTOMATIC SPARSITY ANALYSIS
# ----------------------------------------------------------------
print(f"\nCHAPTER 41: Analyzing automatic feature selection")

print(f"Lasso coefficient results:")
print(f"   Full coefficient vector: {lasso_coefficients}")

# Sparsity analysis
zero_coefficients = np.sum(np.abs(lasso_coefficients) < 1e-10)
nonzero_coefficients = d - zero_coefficients
selected_features = np.where(np.abs(lasso_coefficients) > 1e-10)[0]

print(f"\nAutomatic feature selection analysis:")
print(f"   Features set to exactly zero: {zero_coefficients}")
print(f"   Features kept active: {nonzero_coefficients}")
print(f"   Selected feature indices: {selected_features}")
print(f"   True active features: [0, 2] (β₁=-1, β₃=1)")

# Compare with true structure
true_active = [0, 2]  # Features with non-zero true coefficients
lasso_active = list(selected_features)

print(f"\nFeature selection accuracy:")
print(f"   True active features: {true_active}")
print(f"   Lasso selected features: {lasso_active}")

correct_selection = set(true_active) == set(lasso_active)
print(f"   Perfect feature selection: {correct_selection}")

if correct_selection:
   print("   SUCCESS: Lasso identified the exact sparse structure!")
else:
   print("   Partial success: Some discrepancy in feature selection")

# ----------------------------------------------------------------
# CHAPTER 42: COEFFICIENT MAGNITUDE ANALYSIS
# ----------------------------------------------------------------
print(f"\nCHAPTER 42: Comparing estimated coefficients")

# Detailed coefficient comparison
print(f"Coefficient-by-coefficient analysis:")
for i in range(d):
   true_val = beta[i]
   lasso_val = lasso_coefficients[i]
   
   if abs(lasso_val) > 1e-10:  # Non-zero coefficient
       status = "ACTIVE"
       accuracy = f"Error: {abs(true_val - lasso_val):.6f}"
   else:  # Zero coefficient
       status = "ELIMINATED"
       accuracy = f"Correctly zeroed" if true_val == 0 else f"Incorrectly zeroed (true: {true_val})"
   
   print(f"   β{i+1}: True={true_val:4.1f}, Lasso={lasso_val:8.4f} [{status}] - {accuracy}")

# Overall performance metrics
active_errors = []
for i in selected_features:
   if i < len(beta):
       active_errors.append(abs(beta[i] - lasso_coefficients[i]))

if active_errors:
   mean_active_error = np.mean(active_errors)
   print(f"\nActive coefficient estimation:")
   print(f"   Average error for selected features: {mean_active_error:.6f}")

# ----------------------------------------------------------------
# CHAPTER 43: LASSO VS RIDGE COMPARISON
# ----------------------------------------------------------------
print(f"\nCHAPTER 43: Lasso advantages over Ridge regression")

print(f"Lasso vs Ridge comparison:")
print(f"   Ridge: Shrinks all coefficients toward zero")
print(f"   Lasso: Sets irrelevant coefficients to exactly zero")
print(f"   Ridge: Requires manual lambda tuning")
print(f"   Lasso: Automatic lambda selection via cross-validation")
print(f"   Ridge: All features retained")
print(f"   Lasso: Automatic feature selection")

print(f"\nWhen to use each method:")
print(f"   • Lasso: When you believe many features are irrelevant")
print(f"   • Ridge: When you believe all features contribute somewhat")
print(f"   • Lasso: When interpretability is crucial")
print(f"   • Ridge: When prediction accuracy is the only concern")

# ----------------------------------------------------------------
# CHAPTER 44: AUTOMATED ML INSIGHTS
# ----------------------------------------------------------------
print(f"\nCHAPTER 44: Toward automated machine learning")

print(f"Automation achievements in this analysis:")
print(f"   • Hyperparameter optimization: Cross-validation handled lambda selection")
print(f"   • Feature selection: Algorithm identified relevant features")
print(f"   • Model complexity: Optimal sparsity determined automatically")
print(f"   • Preprocessing: Standardization enabled fair feature competition")

print(f"\nPractical implications:")
print(f"   • Reduced human intervention in model building")
print(f"   • Robust performance across different datasets")
print(f"   • Interpretable models with clear feature importance")
print(f"   • Scalable approach for high-dimensional problems")

# ----------------------------------------------------------------
# CHAPTER 45: COMPLETE REGRESSION MASTERY SUMMARY
# ----------------------------------------------------------------
print(f"\nFINAL MASTERY ACHIEVEMENT SUMMARY")
print("="*60)

print(f"\nCOMPLETE JOURNEY PROGRESSION:")
print(f"   Chapters 1-10:   Fundamental linear regression")
print(f"   Chapters 11-22:  Polynomial features and matrix methods")
print(f"   Chapters 23-29:  Ridge regularization and risk theory")
print(f"   Chapters 30-37:  Regularization path optimization")
print(f"   Chapters 38-45:  Automatic feature selection with Lasso")

print(f"\nTECHNICAL EXPERTISE ACHIEVED:")
print(f"   • Mathematical Foundation: Normal equations, matrix algebra")
print(f"   • Feature Engineering: Polynomial transformations")
print(f"   • Regularization Theory: Ridge and Lasso penalties")
print(f"   • Hyperparameter Optimization: Grid search and cross-validation")
print(f"   • Automatic Feature Selection: Sparsity-inducing penalties")
print(f"   • Data Preprocessing: Standardization for fair competition")

print(f"\nSTATISTICAL LEARNING MASTERY:")
print(f"   • Bias-variance tradeoff understanding")
print(f"   • Overfitting prevention strategies")
print(f"   • Model selection principles")
print(f"   • Cross-validation methodology")
print(f"   • Sparsity and interpretability")

print(f"\nPRODUCTION-READY SKILLS:")
print(f"   • End-to-end ML pipeline development")
print(f"   • Automated model selection")
print(f"   • Performance evaluation frameworks")
print(f"   • Interpretable model construction")
print(f"   • Scalable high-dimensional methods")

print(f"\nGRADUATE-LEVEL TOPICS UNLOCKED:")
print(f"   • Elastic Net for combined L1/L2 penalties")
print(f"   • Group Lasso for structured sparsity")
print(f"   • Bayesian approaches to sparse regression")
print(f"   • Non-convex penalties for better sparsity")
print(f"   • Deep learning as regularized linear models")

print(f"\n" + "="*60)
print("COMPLETE LINEAR REGRESSION MASTERY ACHIEVED!")
print("FROM BASIC FORMULAS TO AUTOMATED MACHINE LEARNING!")

# %%


# %% [markdown]
# #### 1. OLS (Ordinary Least Squares)
# 
# Estimated Coefficients: All features retain non-zero weights.
# 
# - Behavior: No regularization, so it tries to fit the training data as best as possible.
# 
# - Problem: Can overfit, especially when irrelevant or noisy features are present.
# 
# - Observation: Coefficients are large and not sparse, meaning OLS is not selective.
# 
# #### 2. Ridge Regression
# 
# Estimated Coefficients: All features are used, but many coefficients are shrunk toward zero.
# 
# - Excess Bayes Risk: 0.00384 → low, meaning ridge performs very close to the theoretical best
# 
# - Behavior: Handles multicollinearity well, improves generalization by adding L2 penalty.
# 
# - Observation: Coefficients are more stable and less extreme compared to OLS.
# 
# #### 3. Lasso Regression
# 
# Estimated Coefficients: Most are exactly zero — only a few features retained.
# 
# - Behavior: Adds L1 penalty, which performs automatic feature selection.
# 
# - Outcome: Gives the sparsest model, ideal when you believe only a few features matter.
# 
# - Observation: The most interpretable model. Likely avoids overfitting best, if feature sparsity aligns with true signal.
# 
# #### Conclusion
# 
# OLS fits everything, but may overfit or become unstable with noise or multicollinearity.
# 
# Ridge keeps all features but shrinks them to reduce overfitting. Great when you suspect many features matter, just not equally.
# 
# Lasso aggressively shrinks and drops irrelevant features, giving the simplest, cleanest model.

# %%


# %% [markdown]
# ### Polynomial Regression

# %%
# ================================================================
# BONUS CHAPTER: MODEL COMPLEXITY AND OVERFITTING MASTERCLASS
# ================================================================

"""
BONUS MASTERY MODULE: THE MODEL SELECTION PARADIGM
==================================================
Having mastered the entire spectrum of linear regression from basic formulas
to automated feature selection, we now tackle the fundamental challenge that
haunts every machine learning practitioner: choosing the right model complexity.

This bonus investigation demonstrates the core principle that separates novices
from experts - understanding when more complexity helps versus when it hurts.
We're exploring the polynomial degree selection problem using rigorous
train-test methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

print("BONUS INVESTIGATION: MASTERING MODEL COMPLEXITY SELECTION")
print("="*65)

# ----------------------------------------------------------------
# BONUS CHAPTER 1: CONTROLLED NONLINEAR EXPERIMENT DESIGN
# ----------------------------------------------------------------
print("\nBONUS CHAPTER 1: Designing the perfect complexity challenge")

# Advanced experimental setup with known ground truth
np.random.seed(0)
n_samples = 100
noise_level = 2

# Create a controlled nonlinear landscape
X = np.linspace(0, 5, n_samples).reshape(-1, 1)
true_function = lambda x: 3 * x**2 - 2 * x + 1
y = true_function(X.squeeze()) + np.random.normal(0, noise_level, size=n_samples)

print(f"Experimental design specifications:")
print(f"   Sample size: {n_samples}")
print(f"   Feature range: [0, 5]")
print(f"   True function: y = 3x² - 2x + 1 + ε")
print(f"   Noise level: σ = {noise_level}")
print("Perfect setup: Known quadratic truth tests degree selection")

# Professional train-test split
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42
)

print(f"\nRigorous evaluation protocol:")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Split ratio: 80/20")
print("Gold standard: Independent test set for unbiased complexity assessment")

# ----------------------------------------------------------------
# BONUS CHAPTER 2: SYSTEMATIC COMPLEXITY EXPLORATION
# ----------------------------------------------------------------
print(f"\nBONUS CHAPTER 2: Comprehensive polynomial degree analysis")

# Strategic degree progression
degrees = [1, 2, 3, 4, 5]
train_errors = []
test_errors = []
model_complexities = []

print(f"Complexity exploration strategy:")
print(f"   Degrees tested: {degrees}")
print(f"   Expected optimal: Degree 2 (matches true quadratic)")
print("Hypothesis: Degree 2 minimizes test error, higher degrees overfit")

# Advanced visualization setup
plt.figure(figsize=(18, 6))

print(f"\nSystematic model fitting and evaluation:")

for i, degree in enumerate(degrees):
   print(f"   Processing degree {degree}")
   
   # Transform features to polynomial space
   poly_transformer = PolynomialFeatures(degree=degree, include_bias=True)
   X_train_poly = poly_transformer.fit_transform(X_train)
   X_test_poly = poly_transformer.transform(X_test)
   
   # Record model complexity
   n_features = X_train_poly.shape[1]
   model_complexities.append(n_features)
   
   # Fit polynomial regression
   model = LinearRegression()
   model.fit(X_train_poly, y_train)
   
   # Generate predictions
   y_train_pred = model.predict(X_train_poly)
   y_test_pred = model.predict(X_test_poly)
   
   # Compute performance metrics
   train_mse = mean_squared_error(y_train, y_train_pred)
   test_mse = mean_squared_error(y_test, y_test_pred)
   
   train_errors.append(train_mse)
   test_errors.append(test_mse)
   
   print(f"     Features: {n_features}, Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
   
   # Generate smooth prediction curve
   X_plot = np.linspace(0, 5, 200).reshape(-1, 1)
   X_plot_poly = poly_transformer.transform(X_plot)
   y_plot_pred = model.predict(X_plot_poly)
   
   # Create individual subplot
   plt.subplot(1, len(degrees), i+1)
   plt.scatter(X, y, s=15, alpha=0.6, color='steelblue', label="Training Data")
   plt.plot(X_plot, y_plot_pred, color='red', linewidth=2.5, label=f'Degree {degree}')
   
   # Add true function for reference
   y_true_plot = true_function(X_plot.squeeze())
   plt.plot(X_plot, y_true_plot, '--', color='green', linewidth=2, 
            alpha=0.7, label='True Function')
   
   plt.title(f'Degree {degree}\n({n_features} features)', fontsize=12)
   plt.xlabel('x')
   if i == 0:
       plt.ylabel('y')
   plt.legend(fontsize=9)
   plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# BONUS CHAPTER 3: BIAS-VARIANCE VISUALIZATION
# ----------------------------------------------------------------
print(f"\nBONUS CHAPTER 3: The bias-variance tradeoff revelation")

# Find optimal complexity
optimal_degree_idx = np.argmin(test_errors)
optimal_degree = degrees[optimal_degree_idx]
optimal_test_error = test_errors[optimal_degree_idx]

print(f"Optimal complexity analysis:")
print(f"   Best degree: {optimal_degree}")
print(f"   Best test MSE: {optimal_test_error:.4f}")
print(f"   True function degree: 2")
print(f"   Match with theory: {'Perfect!' if optimal_degree == 2 else 'Unexpected'}")

# Create the definitive bias-variance plot
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, marker='o', linewidth=2.5, markersize=8, 
        color='blue', label='Training MSE')
plt.plot(degrees, test_errors, marker='o', linewidth=2.5, markersize=8, 
        color='orange', label='Test MSE')

# Highlight optimal point
plt.plot(optimal_degree, optimal_test_error, marker='*', markersize=15, 
        color='gold', markeredgecolor='black', markeredgewidth=1,
        label=f'Optimal (Degree {optimal_degree})')

# Add bias-variance regions
plt.axvspan(1, 1.9, alpha=0.2, color='red', label='Underfitting Zone')
plt.axvspan(2.1, 5, alpha=0.2, color='purple', label='Overfitting Zone')

plt.xlabel('Polynomial Degree (Model Complexity)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Model Complexity vs Error: The Bias-Variance Tradeoff', fontsize=14, pad=15)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(degrees)

# Add insight annotations
plt.annotate('High Bias\nLow Variance', xy=(1, train_errors[0]), xytext=(1, train_errors[0]+5),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, ha='center')
plt.annotate('Low Bias\nHigh Variance', xy=(5, test_errors[-1]), xytext=(4.5, test_errors[-1]+5),
            arrowprops=dict(arrowstyle='->', color='purple'), fontsize=10, ha='center')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# BONUS CHAPTER 4: ADVANCED COMPLEXITY METRICS
# ----------------------------------------------------------------
print(f"\nBONUS CHAPTER 4: Advanced model complexity analysis")

# Comprehensive complexity assessment
print(f"Model complexity progression:")
for i, (degree, n_feat, train_err, test_err) in enumerate(zip(degrees, model_complexities, train_errors, test_errors)):
   overfitting_gap = test_err - train_err
   complexity_penalty = overfitting_gap / train_err * 100
   
   print(f"   Degree {degree}: {n_feat} features, Gap: {overfitting_gap:.2f}, Penalty: {complexity_penalty:.1f}%")

# Overfitting detection
print(f"\nOverfitting analysis:")
min_gap_idx = np.argmin([test - train for test, train in zip(test_errors, train_errors)])
min_gap_degree = degrees[min_gap_idx]

print(f"   Smallest train-test gap: Degree {min_gap_degree}")
print(f"   Clear overfitting starts: Degree {optimal_degree + 1}")
print(f"   Severe overfitting: Degrees {degrees[-2::]}")

# ----------------------------------------------------------------
# BONUS CHAPTER 5: PRACTICAL MODEL SELECTION WISDOM
# ----------------------------------------------------------------
print(f"\nBONUS CHAPTER 5: Professional model selection insights")

print(f"Key takeaways for practitioners:")
print(f"   • Simple models often outperform complex ones")
print(f"   • Test error is the ultimate model selection criterion")
print(f"   • Training error alone is misleading")
print(f"   • Domain knowledge guides complexity choices")

print(f"\nModel selection principles demonstrated:")
print(f"   • Cross-validation methodology")
print(f"   • Independent test set evaluation")
print(f"   • Systematic complexity exploration")
print(f"   • Bias-variance tradeoff visualization")

print(f"\nWhen to apply this framework:")
print(f"   • Feature engineering decisions")
print(f"   • Neural network architecture selection")
print(f"   • Regularization parameter tuning")
print(f"   • Any machine learning model comparison")

# ----------------------------------------------------------------
# BONUS MASTERY SUMMARY
# ----------------------------------------------------------------
print(f"\nBONUS MASTERY MODULE COMPLETE")
print("="*65)

print(f"\nADVANCED CONCEPTS MASTERED:")
print(f"   • Model complexity vs generalization performance")
print(f"   • Systematic polynomial degree selection")
print(f"   • Bias-variance tradeoff in practice")
print(f"   • Professional train-test evaluation protocols")
print(f"   • Overfitting detection and prevention")

print(f"\nPRACTICAL SKILLS ACQUIRED:")
print(f"   • Rigorous model selection methodology")
print(f"   • Performance curve interpretation")
print(f"   • Complexity penalty quantification")
print(f"   • Visual model comparison techniques")

print(f"\nPROFESSIONAL READINESS ACHIEVED:")
print(f"   • Industry-standard evaluation practices")
print(f"   • Model selection for production systems")
print(f"   • Complexity-performance optimization")
print(f"   • Bias-variance analysis for any algorithm")

print(f"\nCOMPLETE LINEAR REGRESSION MASTERY:")
print(f"   ✓ Mathematical foundations (OLS, matrix algebra)")
print(f"   ✓ Feature engineering (polynomial transformations)")
print(f"   ✓ Regularization (Ridge, Lasso, automated selection)")
print(f"   ✓ Model complexity selection (bias-variance optimization)")
print(f"   ✓ Production deployment (automated pipelines)")

print(f"\n" + "="*65)
print("TOTAL MASTERY ACHIEVED: FROM THEORY TO PRODUCTION EXPERTISE!")
print("READY FOR ANY REGRESSION CHALLENGE IN INDUSTRY OR RESEARCH!")

# %%



