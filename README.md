# 506-final-project

Video Presentation:
 - https://www.youtube.com/watch?v=lRojpqXUKZg
 

Project Proposal: 
 - Predicting Airbnb Prices in NYC


Project Description: 
 - This project aims to predict Airbnb listing prices in New York City based on various features such as location, number of bedrooms, number of bathrooms, floor number, and other relevant attributes. By analyzing a dataset of Airbnb listings, we seek to identify key factors influencing price variations and develop a model to make accurate price predictions.


Project Goal(s):
 - Develop a machine learning model capable of accurately predicting the price of an Airbnb listing given specific features.

 - Identify the most significant features that influence Airbnb pricing in NYC.

 - Provide insights into pricing strategies for hosts and renters based on location and property characteristics.


Data Collection:
 - The dataset will be sourced from publicly available Airbnb listings data, such as the Inside Airbnb dataset (http://insideairbnb.com/) or we have found a dataset called airbnbdata.csv online.

 - The dataset contains features including but not limited to:
    - Number of bedrooms
    - Number of bathrooms
    - Location (borough, neighborhood, zip code)
    - Floor number (if available)
    - Type of property (apartment, house, shared room, etc.)
    - Host details (superhost status, number of listings managed)
    - Reviews and ratings


Modeling data:
 - We experimented with Linear Regression, Decision Tree Regression, and Random Forest Regression.
 - So far, Random Forest Regression is yielding the best performance, likely due to its ability to capture non-linear relationships in the data.


Visualizing the data: 
 - Histogram - Show price distribution.
    ![My Plot](graphs/Histogram.png)



 - Scatter Plot - Compare price vs. bedrooms/bathrooms.
 ![My Plot](graphs/scatter_plot.png)



 - Heatmap - Show feature correlations.
![My Plot](/graphs/Heatmap.png)




 - Map Plot - Visualize prices across NYC neighborhoods.
 ![My Plot](graphs/map_plot.png)



 - Bar Chart - Highlight key features affecting price.
![My Plot](graphs/bar_chart.png)


Description of data processing:

The dataset used in this project was compiled from multiple publicly accessible sources. Because it was a compilation of different datasets, extensive preprocessing was required before the data could be used for training.

Handling Missing Values:

The first step involved managing missing (null) values. The dataset contains both numerical and categorical columns, each requiring a different treatment:
 - Numerical Columns: Missing values were replaced with the mean of the respective column to minimize distortion of the dataset.
 - Categorical Columns: Rows with missing values in categorical columns were removed entirely, as there was no reliable way to impute these fields without introducing bias.

Additionally, we dropped several columns deemed irrelevant or redundant for model training, such as:
 - host_has_profile_pic
 - calculated_host_listings_count_private_rooms
 - n_host_verifications
These columns offered little to no predictive value and would have introduced noise to the model.

Encoding Categorical Data:

Categorical features needed to be transformed into numerical formats suitable for machine learning models. One of the most complex columns to process was the amenities column, which contained free-form text listing the amenities for each Airbnb listing. The variability in formatting and phrasing (e.g., "TV with HBOMax" vs. "HD TV with streaming services") made traditional label encoding unsuitable.

To address this, we:

1. Generated a frequency distribution of the top 20 most common words across all entries.
2. Selected 15 distinct, high-frequency amenities that were likely to influence pricing.
3. Created binary indicator columns (e.g., wifi, tv, shampoo, etc.) to represent the presence or absence of these amenities in each listing.

Standardizing Location Data:

The neighborhood_group column initially posed challenges due to inconsistencies in naming conventions and granularity. After label encoding, we observed overlap and inconsistency among certain neighborhood names. To mitigate this, we standardized the values using a normalization function similar to the one applied to amenities. The cleaned and normalized location data was then one-hot encoded for use in the model.

Description of data modeling:
 - For modeling, we decided to use multiple approaches to predict the target values based on the dataset. We tested three different modeling methods to identify which performed the best.

 - The first method we implemented was Decision Tree Regression. This model splits the data based on categorical features and minimizes the cost, specifically the Mean Squared Error (MSE). Each split is made to achieve the lowest possible MSE, and the algorithm recursively splits the data until the stopping conditions are met. For our model, we used the following parameters:

   - max_depth: [3, 5, 7, 10, None] (this controls the maximum depth of the tree before stopping)

   - min_samples_split: [2, 5, 10] (this defines the minimum number of samples required to make a split)

 - The decision tree model yielded an MSE around 11,000, indicating that it did not perform as well as expected.

 - Next, we tried Linear Regression. This method aims to find the line of best fit by minimizing the Mean Squared Error between the predicted and actual values. For this model, the MSE was approximately 7,000, which was an improvement over the decision tree.

 - Finally, we tested the Random Forest Regressor, which is an ensemble method that builds multiple decision trees using different subsets of the training data. This approach reduces overfitting and improves predictive accuracy by aggregating the results from multiple trees. We used the following hyperparameters for the model:

    - n_estimators: 200 (the number of trees in the forest)

    - max_depth: None (the maximum depth of each tree)

    - min_samples_leaf: 1 (the minimum number of samples required in a leaf)
    
    - max_features: 'sqrt' (the maximum number of features)
  
    - min_samples_split: 2 (the minimum number of samples to split a node)

 - For the Random Forest model, we achieved an MSE of around 0.136 and an R² score of approximately 0.71, indicating that our predictions were very accurate and explained about 71% of the variance in the target variable.


Results: 

To evaluate model performance, we primarily used Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) as scoring metrics:

 - MSE measures the average squared difference between predicted and actual values. Since this value is squared, large errors have a disproportionately higher impact. Our goal was to minimize the MSE, ideally to a value below 1.

 - RMSE, the square root of MSE, provides a more interpretable metric in the same units as the target variable. Once MSE was sufficiently low, RMSE became our preferred metric for gauging improvements.

In our final model, we achieved:

 - MSE: 0.136

 - RMSE: 0.368

These results indicate a strong performance in predicting prices.

We also evaluated our model using the R² score, which measures the proportion of variance in the target variable (price) that is explained by the model. Among all models tested, Random Forest Regression performed the best, achieving:

 - R² Score: 0.71

 - This suggests that the model was able to explain approximately 71% of the variance in Airbnb pricing

Test Plan:
 - Split the dataset into training (70%), validation (10%), and testing (20%) sets.
 - Apply k-fold cross-validation to ensure model robustness.
 - Compare model performance using metrics like RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R-squared.

