## Folder Structure

```
data/
├── raw/         # Contains the original unmodified AmesHousing dataset
├── processed/   # Cleaned dataset after preprocessing and feature engineering
├── train/       # Training set (features and target)
├── test/        # Testing set (features and target)
```

---

## Data Flow

1. **Raw Data** is stored in the `raw/` folder.
2. This raw data is **cleaned and preprocessed** using the `data_wrangling.py` script in the `src/` directory and saved in the `processed/` folder.
3. The cleaned dataset is then **split into training and testing sets** using the `split_data.py` script in `src/`.
4. These train and test sets are used to **train** machine learning models and later to **evaluate** their performance.

---

All transformations are performed via dedicated scripts in the `src/` repo for full reproducibility.