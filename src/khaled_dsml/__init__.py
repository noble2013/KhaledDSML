"""KhaledDSML: educational statistics, ML and big data in Python."""
__version__ = "0.2.0"

from .statistics import (normality_test, shapiro_test, jarque_bera_test,
    independent_t_test, paired_t_test, mann_whitney_test, wilcoxon_test,
    anova_test, kruskal_test, levene_test, chi_square_test, pearson_test,
    spearman_test, ks_two_sample_test, descriptive_summary)
from .quality import (quality_report, missingness_report, iqr_outliers,
    zscore_outliers, fill_missing_numeric, memory_report, optimize_memory,
    distribution_drift_test)
from .ml import (split_data, make_regressor, make_classifier, regression_metrics,
    classification_metrics, cross_validation_summary, cluster_data, pca_summary)
from .bigdata import (generate_transactions, csv_chunk_summary, csv_chunk_missing,
    csv_chunk_monthly, map_reduce_word_count, json_documents_to_table)
from .quizzes import Question, get_questions, grade_quiz
from .timeseries import lagged_features
from .sqltools import sqlite_import_frame, sqlite_group_summary

from .reports import render_test_report, save_test_report, display_test_report

from .ml_evaluation import (
    binary_classification_metrics, roc_curve_table, compare_classifiers,
    render_classification_report, save_classification_report,
    display_classification_report,
)
