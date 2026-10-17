# سجل التغييرات | Changelog

## 0.5.0 — قيد التطوير / Unreleased

- اختبارات افتراضات متعددة المتغيرات Mardia، Box M، Mauchly، Breusch–Godfrey وLevene mean.
- اختبارات الاتجاه الفئوي، G²، CMH، Newcombe، Welch TOST وحساب القوة المسبقة.
- Omega one-factor، ICC2/ICC3، Krippendorff nominal alpha، جودة التضمين، Distance correlation، استقرار العناقيد.
- دعم مدونات Unicode العربية/الإنجليزية: Zipf، TTR، Dunning G²، وفواصل paired bootstrap.
- يحافظ على تقارير Analytics Studio الإبداعية ذات محاذاة اللغة الصحيحة.

## 0.4.0 — قيد التطوير / Unreleased

- `inference_extended`: Welch ANOVA، Brunner-Munzel، Mood median، McNemar، Cochran Q، z للنسب، Wilson، ارتباطات جزئية، أحجام الأثر ونسب المخاطر.
- `psychometrics_extended`: ارتباط البند بالمجموع وتصحيح alpha، split-half، اتفاق المحكّمين، EFA، Parallel analysis، AVE/CR، HTMT وFornell-Larcker.
- `ml_extended`: MCC، κ، PR-AUC، Brier، Log loss، ECE، Calinski-Harabasz، ARI/NMI، فجوات الإنصاف، تقسيم السلاسل الزمني.
- `regression_extended`: White، Goldfeld-Quandt، RESET، Rainbow، Cook's D، Leverage، DFFITS، DFBETAS، Chow، CUSUM، AIC/BIC، Hosmer-Lemeshow، two-way ANOVA.
- `timeseries_extended`: KPSS، Zivot-Andrews، Granger predictivity، Engle-Granger، Johansen، Diebold-Mariano.
- منهجية الاختبارات: توثيق عربي/إنجليزي، وضبط حالات الإدخال، ومقارنة مرجعية مع SciPy وstatsmodels وscikit-learn.
- الإصدار 0.3.0 المنشور يظل دون تعديل. لا نعدّ كل كتالوج 256 بندًا منجزًا.

## 0.3.0 — منشور رسميًا على PyPI / Published

- **Analytics Studio:** تصميم أكاديمي جديد بترويسة متدرجة، بطاقات مؤشرات، تفسير ثنائي اللغة ومحاذاة صحيحة، وثيمات Aurora/Ocean/Amethyst؛ مع SVG اختياري مستمد من البيانات الحقيقية.
- صيغة عرض افتراضية `style="studio"` وإبقاء `style="plain"` للتوافق، مع حفظ HTML صالح للطباعة ودفتر Colab إبداعي.

- اختبار اتجاه Mann–Kendall مع معالجة التكرارات والتفسير بالعربية والإنجليزية.
- اختبار Pettitt لتقدير نقطة تغير واحدة في موقع السلسلة باستخدام قيم احتمالية تقريبية.
- اختبارات Box–Pierce وEngle ARCH-LM لبواقي النماذج الزمنية، بمتطلب statsmodels الاختياري.
- مقاييس جودة التنبؤ MAE وRMSE وMASE وsMAPE باستخدام مرجع موسمي محسوب من عينة التدريب.
- وحدات SHA-256 وبصمة ملفات البيانات وقائمة إصدارات البرامج وملف JSON لإعادة الإنتاج، دون حفظ مسارات شخصية مطلقة.
- حالات اختبار مرجعية تحافظ على ترتيب الزمن وتتحقق من حدود العينات والتوثيق ثنائي اللغة.
- التنفيذ على فرع مستقل؛ لم تُنشر النسخة على PyPI بعد.

## 0.2.0 — منشور رسميًا على PyPI

- دعم Dask/Spark اختياري: CSV بالأجزاء، تجميع الولاية والشهر، وجودة الفقد دون جمع بيانات الخام في pandas.
- محرك Spark محلي واختبارات تكامل فعلية باستخدام Java، وتثبيت اختياري `[dask]` و`[spark]` و`[distributed]`.

- وحدة تدقيق البيانات: تكرارات الصفوف، فحص الفقد، الأعمدة الثابتة، MAD/Mahalanobis، Isolation Forest/LOF، straight-lining، بنفورد، الأرقام النهائية، تداخل مجموعات التدريب/الاختبار.
- وحدة افتراضات جديدة: Anderson–Darling وLilliefors وRuns وDurbin–Watson وKMO وBartlett للكروية وVIF/condition index.
- تقارير تدقيق ثنائية اللغة وتوثيق أكاديمي يحذّر من تفسير مؤشرات الشذوذ بوصفها دليلًا على الاحتيال.
- اختبارات `pytest` لحالات مرجعية وحدود صلاحية بيانات الإدخال.

- تقارير عربية RTL للاختبارات الإحصائية مع مراعاة اتجاه الأرقام.
- تقييم التصنيف الثنائي: ROC-AUC وAverage Precision وF1 والاسترجاع والنوعية والدقة المتوازنة.
- مقارنة النماذج على بيانات اختبار منفصلة دون إعادة تدريب النماذج الأصلية.
- منحنى ROC قابل للرسم وتقرير HTML عربي لتقييم التصنيف.
- عرض مباشر داخل Jupyter وGoogle Colab بدون كتابة HTML يدويًا.
- حفظ التقارير محليًا بصيغة HTML جاهزة للطباعة.
- معالجة النصوص غير الموثوقة والتمييز بين p-value الصغيرة والصفر المعروض بالتقريب.
- اختبارات آلية لتقارير HTML ودليل عربي وأمثلة تعليمية.

## 0.1.0 — منشور رسميًا على PyPI

- اختبارات الاستدلال الإحصائي، وجودة البيانات، وتقييم نماذج تعلم الآلة.
- أدوات محلية لمعالجة ملفات CSV بالأجزاء، وأمثلة اقتصادية تعليمية.
- التقويم بأسئلة اختيار متعدد وتصحيح آلي، ودفاتر Jupyter عربية.
- GitHub Actions للتحقق المستمر والنشر التجريبي والرسمي.
