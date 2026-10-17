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

from .descriptive import (
    describe_numeric, describe_categorical, descriptive_table, frequency_table,
    weighted_mean, gini_coefficient, covariance_correlation, interpret_descriptive,
    render_descriptive_report, save_descriptive_report, display_descriptive_report,
)

from .catalog_tests import (
    one_sample_t_test, dagostino_normality_test, bartlett_test,
    fligner_test, kendall_test, fisher_exact_test, chi_square_gof_test,
    friedman_test, sign_test, adjust_pvalues, cohen_d, cliffs_delta,
    bootstrap_mean_ci, permutation_mean_test,
)
from .ml_evaluation import interpret_classification_metrics

from .audit import (
    duplicate_rows_report, missing_pattern_report, column_diagnostics,
    modified_z_outliers, mahalanobis_outliers, ml_outliers,
    survey_straightlining, benford_first_digit_test, terminal_digit_test,
    train_test_overlap, audit_dataset, render_audit_report, save_audit_report,
)
from .assumptions_extra import (
    anderson_darling_normality, lilliefors_normality, runs_test,
    durbin_watson_statistic, kmo_sampling_adequacy,
    bartlett_sphericity_test, condition_number_vif,
)

from .distributed import (
    dask_read_csv, dask_group_summary, dask_csv_summary,
    dask_missingness_report, dask_monthly_summary,
    create_spark_session, spark_read_csv, spark_group_summary,
    spark_csv_summary, spark_missingness_report, spark_monthly_summary,
)
