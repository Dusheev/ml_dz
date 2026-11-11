# Salary prediction app

Streamlit front-end for the model from `job_postings_fixed.ipynb`.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## Files
- `app.py` — UI (form, prediction, what-if charts)
- `salary_core.py` — feature building identical to the notebook (section 7) + prediction
- `best_salary_model.joblib` — trained model (replace with yours after re-running the notebook)
- `data-job-postings.csv` — used for dropdown options and "similar postings" context
- `salary_model_comparison.csv` — used to show the model's typical error (MAPE)

If you retrain with CatBoost, uncomment `catboost` in requirements.txt and
make sure the scikit-learn version matches the one in your notebook environment.
