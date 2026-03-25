# Day 3 — Data Visualisation + EDA

**Date:** [25/03/26]
**Time invested:** ~7 hours
**Kaggle Notebook:** [https://lnkd.in/giC-R-My]

## What I Built
- `matplotlib_basics.py` — Line, bar, scatter, histogram charts from scratch
- `seaborn_basics.py` — Statistical charts using Titanic data
- `titanic_eda.py` — Complete EDA on 891-passenger Titanic dataset

## Key Findings From The Data
1. Women survived at 74% vs men at 19% — gender was the strongest predictor
2. First class survival (63%) vs Third class (24%) — class determined access to lifeboats
3. Small families (2-4 members) had higher survival than solo travellers
4. Cabin column (77% missing) is unusable for ML without heavy engineering
5. Age and Fare alone are weak predictors but interact with class and sex

## Charts Produced
13 charts saved to outputs/ covering:
- Missing value analysis
- Target variable distribution  
- Categorical features vs survival
- Numerical features vs survival
- Correlation heatmap

## ML Readiness Conclusion
- **Use:** Sex, Pclass, Age, Fare, family_size
- **Drop:** Cabin, Name, Ticket
- **Fix:** Age (22% missing — median imputation by class)
- **Problem type:** Binary classification

## Skills Gained
- Matplotlib figure/axes architecture
- Seaborn statistical plotting
- Systematic EDA workflow
- Reading and interpreting correlation matrices
- Identifying ML-ready vs unusable features

## Kaggle Notebook
[Titanic EDA](https://kaggle.com/code/arindambuilds/titanic-eda-week01-day03)