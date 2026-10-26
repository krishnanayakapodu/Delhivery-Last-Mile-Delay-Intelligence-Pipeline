\# Delhivery Last-Mile Delay Intelligence Pipeline



This project uses synthetic shipment data to analyze delivery delays, predict late parcels, and group hubs by observed performance.



\## How to run



Windows Git CMD commands from a fresh clone:



```cmd

git clone https://github.com/krishnanayakapodu/Delhivery-Last-Mile-Delay-Intelligence-Pipeline.git

cd Delhivery-Last-Mile-Delay-Intelligence-Pipeline

py -3 -m venv .venv

.venv\\Scripts\\python.exe -m pip install -r requirements.txt

.venv\\Scripts\\python.exe generate\_data.py

.venv\\Scripts\\python.exe part\_a\_clean\_eda.py

.venv\\Scripts\\python.exe part\_b\_regression.py

.venv\\Scripts\\python.exe part\_c\_classification.py

.venv\\Scripts\\python.exe part\_d\_segments.py

.venv\\Scripts\\python.exe package\_model.py

.venv\\Scripts\\python.exe predict\_delay.py

```



`package\_model.py` regenerates `models/delay\_model.pkl`. The committed pickle is reliable only with the scikit-learn version pinned in `requirements.txt`. If the grader has a different version, it should rerun `package\_model.py`.



\## Data findings — Part A



\- Cleaning removed 24 duplicate rows, leaving 1,200 shipments. The modelling table has 278 delayed parcels (23.17%) and no missing model features or target values.

\- Delay rates were 48.97% for Express, 20.56% for Standard, and 8.57% for Economy. Rates rose from 2.19% for parcels up to 200 km to 85.23% for parcels over 600 km.

\- The strongest positive correlations with delivery hours were promised hours (0.8372), distance (0.4013), and number of stops (0.3247).



See `outputs/part\_a\_checks.csv`, `outputs/part\_a\_delay\_by\_group.csv`, and `outputs/part\_a\_correlations.csv`.



\## Model results (regression) — Part B



\- The mean baseline had validation R² of −0.0212. Linear regression won validation with RMSE of 4.8912 hours; its test R² was 0.9493.

\- Tree depth 6 had the lowest validation RMSE. Depth 1 had high training and validation errors; the unlimited tree had 0.0000 training RMSE and 7.7890 validation RMSE, indicating overfitting.

\- Gradient boosting learning rate 0.1 performed best. At 0.01, learning was too slow; at 1.0, it overshot.

\- Cross-validation ranked gradient boosting above linear regression (mean R² 0.9352 vs 0.9261), but linear regression remained the choice because it had lower validation RMSE.



See the `outputs/part\_b\_\*.csv` files.



\## Model results (classification) — Part C



\- The blind baseline had 0.7667 accuracy but found no delayed parcels.

\- The top seven features were distance, hub tier, hub capacity, stops, promised hours, rain, and sale-event flag. The five noise columns contributed 22.45% of the 17-feature forest importance.

\- Selecting seven features gave the best validation F1 (0.8333). The untuned forest beat grid search (0.7750) and randomized search (0.7848) on validation F1.

\- Threshold 0.5 was the highest tested threshold with validation recall of at least 0.80. On test data, precision was 0.6889, recall 0.7561, and F1 0.7209. The test set had 41 delayed parcels.



See the `outputs/part\_c\_\*.csv` files.



\## Operations recommendation — Part D



The Escalate hubs are HUB13 Nagpur (0.46), HUB15 Kochi (0.36), HUB17 Patna (0.46), HUB18 Ranchi (0.38), HUB19 Guwahati (0.62), HUB20 Raipur (0.48), HUB21 Bhubaneswar (0.40), HUB23 Siliguri (0.44), and HUB24 Agartala (0.54).



Tier mismatches are HUB07 Kolkata (Tier 1, Watch), HUB13 Nagpur (Tier 2, Escalate), HUB15 Kochi (Tier 2, Escalate), and HUB22 Dehradun (Tier 3, Watch). Use observed segment performance when selecting a playbook; tier alone can be misleading.



The new-parcel batch had 8 of 48 parcels flagged: 6 for hub-manager escalation and 2 for proactive customer calls.



K=3 was retained because operations requires three playbooks and its smallest cluster has 7 hubs. Although the separation ratio rises from 5.3928 at K=3 to 6.3047 at K=4 and 6.8866 at K=5, K=4 has a 4-hub cluster and K=5 has clusters of 4, 3, and 2 hubs.



See `outputs/part\_d\_separation.csv`, `outputs/segment\_profile.csv`, `outputs/tier\_vs\_segment\_mismatch.csv`, and `outputs/new\_shipment\_actions.csv`.



\## Production plan



\- \*\*Accuracy:\*\* review `outputs/part\_b\_metrics.csv` and `outputs/part\_c\_final\_test.csv`.

\- \*\*Reliability:\*\* `outputs/part\_d\_reload\_check.csv` checks that the packaged model reloads consistently.

\- \*\*Reproducibility:\*\* fixed random seeds and pinned package versions are used.

\- \*\*Consistency and maintainability:\*\* `src/shipment\_prep.py` shares preprocessing between training and scoring.

\- \*\*Monitoring:\*\* investigate if weekly delay share exceeds 30%, if more than 25% of parcels are flagged, or if recall on known outcomes falls below 0.70.

\- Retrain quarterly, or sooner if monitoring shows sustained problems. Compare the 15–28 September sale period with similar sale periods, not ordinary weeks. Watch for changes in service mix, distance, stops, hub characteristics, and sale-event share.



\## Limitations



\- The data is synthetic and may not represent real logistics operations.

\- The test set contains only 41 delayed parcels, so classification metrics may be sensitive to a small number of cases.

\- The threshold was selected using one validation split.

\- The model lacks real-time information such as traffic, sudden weather changes, staffing, and vehicle disruptions.

\- Hub segments depend on the selected profile features and clustering settings.

