# Model comparison — test set (20% hold-out)

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Fit time |
|---|---|---|---|---|---|---|---|
| **Random Forest** | 0.6777 | 0.2167 | 0.6780 | 0.1854 | 0.5557 | 0.2781 | 15s |
| Keras MLP (neural net) | 0.6750 | 0.2126 | 0.6009 | 0.1695 | 0.6609 | 0.2698 | 19s |
| XGBoost | 0.6709 | 0.2154 | 0.6570 | 0.1786 | 0.5764 | 0.2727 | 4s |
| Logistic Regression | 0.6684 | 0.2077 | 0.6387 | 0.1745 | 0.5997 | 0.2703 | 4s |
| LogReg + SMOTE | 0.6535 | 0.1996 | 0.6284 | 0.1662 | 0.5804 | 0.2585 | 7s |
