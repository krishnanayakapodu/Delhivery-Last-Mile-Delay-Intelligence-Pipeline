PART A:

## Data findings



1\. Service-Level Delay Analysis

&#x20;  - Express shipments recorded the highest delay rate (48.97%), followed by Standard (20.56%) and Economy (8.57%).

&#x20;  - The observed ordering of delay rates is Express > Standard > Economy.

&#x20;  - Source: outputs/part\_a\_delay\_by\_group.csv



2\. Hub Tier Impact

&#x20;  - Delay rates increased from 4.00% for Tier 1 hubs to 44.00% for Tier 3 hubs.

&#x20;  - This represents a 40 percentage-point increase across hub tiers.

&#x20;  - Source: outputs/part\_a\_delay\_by\_group.csv



3\. Distance-Based Delay Distribution

&#x20;  - Delay rates by distance band were:

&#x20;    - 0-200 km: 2.19%

&#x20;    - 201-400 km: 13.68%

&#x20;    - 401-600 km: 43.64%

&#x20;    - 601+ km: 85.23%

&#x20;  - The observed delay rate increases monotonically with shipment distance.

&#x20;  - Source: outputs/part\_a\_delay\_by\_group.csv



4\. Correlation Analysis

&#x20;  - Variables with the strongest positive correlation with delivery hours were:

&#x20;    - promised\_hours: +0.8372

&#x20;    - distance\_km: +0.4013

&#x20;    - n\_stops: +0.3247

&#x20;  - Among the evaluated features, promised\_hours is the most strongly correlated with delivery hours.

&#x20;  - Source: outputs/part\_a\_correlations.csv



5\. Log Transformation Assessment

&#x20;  - The imputed declared\_value\_inr variable had a mean of INR 1,533.05 and a median of INR 1,121.50, resulting in a mean-median gap of INR 411.55.

&#x20;  - After log transformation, the mean and median were 7.0277 and 7.0224, respectively, reducing the gap to 0.0053 log units.

&#x20;  - The reduced mean-median gap is consistent with the compression of high-value observations in the upper tail of the distribution.

&#x20;  - Source: outputs/part\_a\_checks.csv





PART B:

## Regression findings



1\. Model Selection

&#x20;  - Among the evaluated models, Linear Regression achieved the lowest validation RMSE of 4.8912 hours and was selected as the final model.



2\. Baseline Performance

&#x20;  - The baseline model produced an R² score of -0.0212, indicating that predicting the average delivery time provides little predictive value for this dataset.



3\. Decision Tree Depth Comparison

&#x20;  - A maximum depth of 6 produced the best validation performance.

&#x20;  - The depth-1 tree showed high training and validation errors, indicating underfitting.

&#x20;  - The unrestricted tree achieved zero training RMSE but a validation RMSE of 7.7890, suggesting overfitting.



4\. Learning Rate Evaluation

&#x20;  - A learning rate of 0.1 provided the best results among the tested values.

&#x20;  - The 0.01 learning rate learned too slowly with 100 estimators, while 1.0 was too aggressive and overshot, leading to poorer validation performance.



5\. Cross-Validation Results

&#x20;  - Gradient Boosting achieved a slightly better average cross-validation score than Linear Regression.

&#x20;  - However, final model selection was based on validation RMSE, where Linear Regression performed better.

&#x20;  - The selected Linear Regression model achieved a test R² score of 0.9493.







PART C:



\## Model results (classification)



1\. Baseline Performance

&#x20;  - The blind baseline achieved an accuracy of 0.7667 by predicting the majority class for every validation parcel.

&#x20;  - While the accuracy appears relatively high, the model does not distinguish between delayed and non-delayed parcels, making it of limited practical value for delay prediction.



2\. Feature Importance Analysis

&#x20;  - The seven selected features were:

&#x20;    - distance\_km

&#x20;    - hub\_tier

&#x20;    - hub\_daily\_capacity

&#x20;    - n\_stops

&#x20;    - promised\_hours

&#x20;    - rain\_flag

&#x20;    - sale\_event\_flag

&#x20;  - These features captured most of the predictive signal in the dataset.

&#x20;  - The combined Random Forest importance assigned to the noise features was 0.2245 (22.45%), indicating that the majority of feature importance was concentrated in the selected signal variables.



3\. Feature Selection and Tuning

&#x20;  - Seven features (k=7) were selected because they provided the best validation performance during feature selection.

&#x20;  - The untuned Random Forest trained on the selected features achieved a validation F1 score of 0.8333.

&#x20;  - This outperformed both Grid Search tuning (0.7750) and Randomized Search tuning (0.7848), so the untuned model was retained as the final classifier.



4\. Classification Threshold Selection

&#x20;  - The operational requirement prioritized recall when selecting the classification threshold.

&#x20;  - A threshold of 0.5 achieved a validation recall of 0.8333, compared with 0.7619 at a threshold of 0.6.

&#x20;  - Threshold 0.5 was the highest tested threshold with validation recall of at least 0.80 and was therefore selected for the final model.



5\. Test Set Performance

&#x20;  - On the test set, the final model achieved precision of 0.6889, recall of 0.7561, and an F1 score of 0.7209.

&#x20;  - Test recall was below validation recall (0.8333). The test set had only 41 delayed parcels, so its recall estimate may be sensitive to a few parcels.

