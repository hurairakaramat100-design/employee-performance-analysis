# Employee Performance Analysis using Python

This beginner-level project analyzes employee performance using working hours, training hours, and experience.

The goal is to understand the data and identify relationships between these factors and the employee performance score before building a Machine Learning model.

## Dataset

The dataset contains **30 sample employee records** with the following columns:

- `Employee_ID` – Unique employee identifier
- `Working_Hours` – Working hours per week
- `Training_Hours` – Training hours completed
- `Experience_Years` – Years of work experience
- `Performance_Score` – Employee performance score

> **Note:** The dataset is synthetic and created for educational/project purposes.

## Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Analysis Performed

1. Loaded and inspected the dataset
2. Checked data types
3. Checked missing values
4. Checked duplicate records
5. Calculated basic statistics
6. Compared working hours with performance
7. Compared training hours with performance
8. Compared experience with performance
9. Created a correlation matrix
10. Summarized the main findings

## Project Structure

```text
employee-performance-analysis-python/
│
├── employee_performance.csv
├── Employee_Performance_Analysis.ipynb
├── README.md
├── CONCLUSION.md
│
└── graphs/
    ├── working_hours_vs_performance.png
    ├── training_hours_vs_performance.png
    ├── experience_vs_performance.png
    └── correlation_matrix.png
```

## Key Results

- Average performance score: **84.63**
- Average working hours: **40.97 hours/week**
- Average training hours: **8.43 hours**
- Average experience: **8.57 years**
- Highest performance score: **100**
- Lowest performance score: **67**
- Strongest positive relationship with performance in this sample: **Experience_Years** (correlation ≈ **0.72**)

## How to Run

1. Download or clone this repository.
2. Open `Employee_Performance_Analysis.ipynb` in Jupyter Notebook or Google Colab.
3. Make sure `employee_performance.csv` is in the same folder.
4. Run the notebook cells from top to bottom.

## Conclusion

The analysis suggests that working hours, training hours, and experience are positively related to employee performance in this sample. These findings provide a useful starting point for a future Machine Learning regression model.

However, correlation does not prove that one factor directly causes higher performance. Real HR decisions would require a larger, real-world dataset and careful consideration of additional factors.

## Project Level

**ML Python Basic – Beginner Level**
