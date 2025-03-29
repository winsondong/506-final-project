# 506-final-project

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
 - We first read the data then start cleaning it by dropping unnecessary columns. We dropped columns pertaining to host information that does not assist in the predictive model such as host’s bio, profile photo, etc.  We then checked for missing values in columns, then filled them with values in numeric columns being replaced by the mean of the rest of the values in the column, and values in categorical columns by dropping the row.
 - The next step of processing includes label encoding — such as changing boolean values in columns to binary, or strings to integer representations. Amenities have also been split such that each amenity offered is its own independent feature with values true or false then get converted to binary. 


Description of data modeling:
 - For modeling, we decided to use multiple approaches to predict the target values based on the dataset. We tested three different modeling methods to identify which performed the best.

 - The first method we implemented was Decision Tree Regression. This model splits the data based on categorical features and minimizes the cost, specifically the Mean Squared Error (MSE). Each split is made to achieve the lowest possible MSE, and the algorithm recursively splits the data until the stopping conditions are met. For our model, we used the following parameters:

   - max_depth: [3, 5, 7, 10, None] (this controls the maximum depth of the tree before stopping)

   - min_samples_split: [2, 5, 10] (this defines the minimum number of samples required to make a split)

 - The decision tree model yielded an MSE around 11,000, indicating that it did not perform as well as expected.

 - Next, we tried Linear Regression. This method aims to find the line of best fit by minimizing the Mean Squared Error between the predicted and actual values. For this model, the MSE was approximately 7,000, which was an improvement over the decision tree.

 - Finally, we tested the Random Forest Regressor, which is an ensemble method that builds multiple decision trees using different subsets of the training data. This approach reduces overfitting and improves predictive accuracy by aggregating the results from multiple trees. We used the following hyperparameters for the model:

    - n_estimators: [100, 200] (the number of trees in the forest)

    - max_depth: [10, 20] (the maximum depth of each tree)

    - min_samples_split: [2, 5] (the minimum number of samples required to split a node)

 - For the Random Forest model, we achieved an MSE of around 5,000 and an R² score of approximately 0.64, indicating that it explained about 64% of the variance in the target variable.


Results: 
 - We evaluated three different regression models on our Airbnb pricing dataset:
      - Linear Regression achieved a Mean Squared Error (MSE) of approximately 9,000, but struggled to capture complex patterns in the data.

      - Decision Tree Regression resulted in a higher MSE of 11,000, with a low accuracy score of 0.04, indicating poor generalization.

      - Random Forest Regression performed the best, with a significantly lower MSE of 5,000 and an R² score of 0.64, suggesting that it was able to explain 64% of the variance in the data.

 - Based on these preliminary results, we plan to continue optimizing and refining the Random Forest model for our final evaluation.


Test Plan:
 - Split the dataset into training (70%), validation (10%), and testing (20%) sets.
 - Apply k-fold cross-validation to ensure model robustness.
 - Compare model performance using metrics like RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R-squared.

