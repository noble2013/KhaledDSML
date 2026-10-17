"""KhaledDSML: educational statistics, ML and big data in Python."""
__version__ = "0.10.0.dev0"

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

from .timeseries_advanced import (
    mann_kendall_trend_test, pettitt_change_point_test,
    box_pierce_test, arch_lm_test, mean_absolute_scaled_error,
    forecast_accuracy_report,
)
from .reproducibility import (
    sha256_file, software_versions, reproducibility_manifest,
    save_reproducibility_manifest, verify_reproducibility_manifest,
)

from .bilingual_display import (
    render_bilingual_result, display_bilingual_result, save_bilingual_result,
    render_bilingual_results, display_bilingual_results, save_bilingual_results,
)

from .visual_reports import (
    render_analysis_dashboard, display_analysis_dashboard, save_analysis_dashboard,
)

# Extended academic catalogue — v0.4 (optional heavy engines stay lazily imported)
from .inference_extended import (
    welch_anova_test, mood_median_test, brunner_munzel_test,
    mcnemar_test, cochran_q_test, one_proportion_z_test,
    two_proportions_z_test, wilson_proportion_interval,
    point_biserial_test, partial_correlation_test, cramers_v,
    phi_coefficient, fisher_z_correlation_test, glass_delta,
    eta_squared, omega_squared, cohen_w, contingency_risk_measures,
)
from .psychometrics_extended import (
    item_total_statistics, split_half_reliability, test_retest_correlation,
    cohen_kappa_agreement, fleiss_kappa_agreement, factor_scree_table,
    parallel_analysis, exploratory_factor_analysis, convergent_validity,
    htmt_ratio, fornell_larcker_matrix, harman_single_factor,
)
from .ml_extended import (
    classification_diagnostics, calibration_summary, detailed_regression_metrics,
    cluster_validation_metrics, binary_group_fairness, time_series_split_indices,
    repeated_cv_indices, model_overfit_gap,
)
from .regression_extended import (
    regression_white_test, regression_goldfeld_quandt_test,
    regression_reset_test, regression_rainbow_test, regression_influence_table,
    chow_structural_break_test, regression_cusum_test, model_information_criteria,
    logistic_hosmer_lemeshow, glm_overdispersion_diagnostic, two_way_anova,
)
from .timeseries_extended import (
    kpss_stationarity_test, zivot_andrews_break_test,
    granger_predictivity_tests, engle_granger_cointegration_test,
    johansen_cointegration_test, diebold_mariano_test,
)

from .research_workflow import (
    compare_groups, research_quality_gate, apa_style_test_summary,
)

# Academic methods completion — v0.5
from .assumptions_multivariate import (
    mardia_multivariate_normality, box_m_covariance_equality,
    mauchly_sphericity_test, breusch_godfrey_test, levene_mean_test,
)
from .inference_advanced import (
    cochran_armitage_trend_test, g_test_independence,
    cochran_mantel_haenszel_test, newcombe_two_proportions_interval,
    welch_tost_equivalence, two_group_power_analysis,
)
from .psychometrics_dimension import (
    omega_single_factor, intraclass_correlation,
    krippendorff_nominal_alpha, trustworthiness_embedding,
    distance_correlation, bootstrap_cluster_stability,
)
from .nlp_corpus import (
    corpus_tokens, lexical_diversity, zipf_frequency_analysis,
    dunning_corpus_keyness, paired_bootstrap_score_difference,
)

# Post-hoc and correlated ROC comparison (v0.6)
from .posthoc_comparisons import (
    tukey_hsd_comparisons, games_howell_comparisons, dunn_posthoc,
    dunnett_control_comparisons, nemenyi_friedman_posthoc,
)
from .roc_comparison import (
    delong_auc_comparison, stratified_paired_auc_bootstrap,
)

# Advanced optional PySpark / Dask engines — v0.7
from .distributed_analytics import (
    spark_missingness_detailed, spark_duplicate_summary,
    spark_numeric_profile, spark_time_holdout,
    spark_fit_logistic_classifier, spark_binary_classification_metrics,
    dask_duplicate_summary, dask_numeric_profile, dask_time_holdout,
)

# Advanced statistical mediation/moderation, v0.8 (statsmodels remains optional)
from .mediation_moderation import (
    mediation_paths, sobel_mediation_test, bootstrap_mediation_interval,
    moderation_interaction_test, simple_slopes_table, johnson_neyman_region,
)

# Missing-data methodology — v0.9
from .missing_data_advanced import (
    little_mcar_test, missingness_association_screen,
    multiple_impute_train_test_numeric, rubin_pool_scalar,
)

# Research methods under development for v0.10.0 (not yet on PyPI)
from .multivariate_normality import henze_zirkler_normality_test
from .missing_data_sensitivity import mnar_delta_sensitivity
