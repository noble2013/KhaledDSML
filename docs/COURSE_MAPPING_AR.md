# خريطة الوظائف حسب المقاييس — KhaledDSML 0.1.0

## 1. علم البيانات وتعلم الآلة — السنة الثالثة ليسانس

| المحور | الوظائف المنفذة حاليًا | المتابعة المطلوبة |
|---|---|---|
| مدخل للبيانات | `quality_report`, `generate_transactions`, أمثلة الاقتصاد | نظرية المفاهيم في الدروس |
| قراءة وتنظيف | `missingness_report`, `fill_missing_numeric`, `iqr_outliers`, `zscore_outliers` | التعويض المتقدم عند الحاجة |
| إحصاء وصفي واستدلالي | `descriptive_summary`, `shapiro_test`, `jarque_bera_test`, `independent_t_test`, `paired_t_test`, `anova_test`, `kruskal_test`, `chi_square_test`, `pearson_test`, `spearman_test`, `levene_test` | التحقق من الفرضيات بحسب تصميم الدراسة |
| تقسيم وتسرّب البيانات | `split_data`, `make_regressor`, `make_classifier` (Pipeline) | مراجعة تسرب الميزات المرتبطة بالهدف |
| الانحدار | Linear Regression, Ridge, Lasso, Random Forest, Decision Tree؛ `regression_metrics` | تفسير المعاملات والتشخيص المتعمق |
| التصنيف | Logistic Regression, KNN, Decision Tree, Random Forest؛ `classification_metrics` | ROC-AUC والمعايرة في الإصدارات التالية |
| التجميع | K-means, Hierarchical، Silhouette، Davies–Bouldin، PCA | خوارزميات تجميع أخرى |
| مقارنة النماذج | `cross_validation_summary` | بحث المعاملات الشبكي وتوصيات عامة |
| التقويم | `get_questions`, `grade_quiz` | أسئلة وتمارين إضافية لكل TD |

## 2. تحليل البيانات الضخمة — السنة الثانية ماستر اقتصاد كمي

| المحور | الوظائف المنفذة حاليًا | حدود الإصدار الحالي |
|---|---|---|
| TD1 توليد وفحص مليون معاملة | `generate_transactions`, `quality_report`, `missingness_report` | توليد DataFrame كامل يحتاج RAM |
| TD1 التنظيف والتجميع | `fill_missing_numeric`, `csv_chunk_summary`, `csv_chunk_monthly` | معالجات الأجزاء تتطلب ملف CSV |
| TD1 تحسين الذاكرة | `memory_report`, `optimize_memory`, `csv_chunk_missing` | تحجيم الفاصلة العائمة قد يغير الدقة وهو غير مفعل افتراضيًا |
| MapReduce | `map_reduce_word_count` | مثال محلي **غير موزّع** |
| قواعد بيانات وNoSQL | `json_documents_to_table`, `sqlite_import_frame`, `sqlite_group_summary` | يتوفر محول Spark اختياري للتجميع فقط؛ اتصال MongoDB/Redis غير متاح |
| تنظيف وإعداد خصائص | `quality`, `ml.make_classifier`, `ml.make_regressor` | تتوفر محولات Dask وSpark الاختيارية؛ الاختبارات الحالية تغطي المعالجة المحلية |
| Lasso/Ridge والغابات | `make_regressor`, `make_classifier`, `cross_validation_summary` | حجم بيانات التدريب محكوم بذاكرة الجهاز |
| تنقيب النصوص | `map_reduce_word_count` | التحليل الدلالي/المشاعر المتقدم مؤجل |
| الاقتصاد القياسي والتسلسل الزمني | `econometrics.adf_test`, `ljung_box_test`, `vif_table`, `breusch_pagan_test` | `lagged_features` متاحة؛ لم تُنفذ نماذج التنبؤ الآني الكاملة |
| الخصوصية والحوكمة | تمارين التقويم | تتطلب سياسات وحوكمة مؤسسية خارج نطاق المكتبة |

## 3. اختبارات إضافية

- `stats`: Shapiro–Wilk، Jarque–Bera، Welch، Student، T المزدوج، Mann–Whitney، Wilcoxon، ANOVA، Kruskal–Wallis، Levene، Chi-square، Pearson، Spearman، KS لعينيّتين.
- `econometrics`: ADF، Ljung–Box، Breusch–Pagan، VIF، Cronbach alpha.
- ليست كل هذه الاختبارات مطلوبة في كل مقياس، بل إضافات مساندة للتحليل.
- لا تشمل النسخة 0.1.0 اختبارات Little MCAR أو EFA/CFA أو PLS-SEM، لأن تقديمها بطريقة صحيحة يتطلب طبقة منهجية مستقلة وحالات تحقق معيارية.
- مستندات المقياسين التي استُخدمت للتصميم هي مواد بيداغوجية مُعدة سابقًا، وليست بطاقة رسمية نهائية؛ **لا يجوز وصف المكتبة بأنها تغطي جميع الاختبارات المقررة رسميًا دون مقارنة ببطاقة التكوين المعتمدة**.

## 4. ملاحظات على التفسير

- `p_value < alpha` يعني رفض H0 في شروط الاختبار ولا يثبت الأهمية العملية أو السببية.
- الاختبارات المعلمية تفترض شروطًا متنوعة (استقلالية، اعتدالية أخطاء أو عينات، تجانس التباين حسب الاختبار).
- نماذج التنبؤ تستلزم فصل تدريب/اختبار؛ في البيانات الزمنية يفضّل التقسيم الزمني بدل التقسيم العشوائي.
- الأرقام الاصطناعية لا تثبت سلوك السوق الجزائري.
