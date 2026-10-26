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

