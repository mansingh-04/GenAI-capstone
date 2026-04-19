# Cleaned Dataset Information

## Description

The cleaned dataset is generated after running the preprocessing pipeline (`preprocess.py`).  

It contains:
- Cleaned and normalized text
- Duplicate removal
- Stop-word filtering (negation preserved)
- Lemmatized tokens
- Binary labels (0 = Fake, 1 = Real)

The final output file:
cleaned_dataset.csv

---

## How to Generate

To generate the cleaned dataset:

1. Download the raw dataset (see data/raw/raw_dataset.md)
2. Place `Fake.csv` and `True.csv` inside:
   data/raw/
3. Run:

   python preprocess.py

The cleaned dataset will be generated automatically.

---

## Optional Download (If Required)

If you prefer downloading the preprocessed dataset directly, use the link below:

https://drive.google.com/file/d/1qXtz_upHWhXLBJGEtvX_XjtJkFayvy2W/view?usp=sharing

---

## Important Note

The cleaned dataset is not stored in this repository to:
- Avoid large file storage
- Ensure reproducibility
- Maintain clean repository structure

Always regenerate the cleaned dataset before training the model.
