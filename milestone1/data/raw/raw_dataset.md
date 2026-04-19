# Raw Dataset Information

## Dataset Name
ISOT Fake News Dataset

## Description
This project uses the **ISOT Fake News Dataset**, which contains labeled news articles categorized as real or fake. The dataset is provided in two separate CSV files:

- **True.csv** – Contains legitimate news articles.
- **Fake.csv** – Contains fabricated or misleading news articles.

Each file includes the following columns:
- `title`
- `text`
- `subject`
- `date`

For this project, the `title` and `text` columns are combined during preprocessing to form a single `content` field used for model training.

---

## Why Dataset Is Not Stored in Repository

The dataset files are not stored directly in this repository due to **GitHub file size limitations**.  
To maintain reproducibility, download links are provided below.

---

## Download Links

- **True.csv**  
  https://drive.google.com/file/d/1Go2-9M1vf1g29s0rn0k7op-hxTWNNtIE/view?usp=sharing  

- **Fake.csv**  
  https://drive.google.com/file/d/1S6o74YUNemtutBl0Vm3TDYSZrt7SVQHb/view?usp=sharing  

---

## Setup Instructions

1. Download both CSV files using the links above.
2. Place them inside the following directory:
data/raw/
3. Ensure your folder structure looks like this:
data/
└── raw/
├── True.csv
└── Fake.csv
4. Run the preprocessing script to generate the cleaned dataset.
---

## Important Notes

- Do not modify the original dataset files.
- All preprocessing is handled by `preprocess.py`.
- The cleaned dataset will be generated separately inside the appropriate directory.
- Ensure both files are correctly placed before running model training.

---

## Reproducibility

Following these steps ensures that the project can be reproduced locally without requiring large dataset files to be stored directly in the repository.
