# دليل إصدار KhaledDSML 0.10 — الإصدار التجريبي والنشر الآمن

**المؤلف:** الأستاذ الدكتور خالد علي — جامعة الشهيد حمه لخضر، الوادي.

## حالة المشروع

| العنصر | الحالة |
|---|---|
| الإصدار الرسمي المتاح سابقًا على PyPI | `0.9.0` |
| إصدار المصدر قيد التحضير | `0.10.0rc1` — Release Candidate 1 |
| مستودع المصدر | https://github.com/noble2013/KhaledDSML |
| مجموع بنود الكتالوج | 256 بندًا |
| البنود المرتبطة بواجهات مرشحة | 256، بينها **بديلان غير مكافئين للطريقة الأصلية** (#85 و#117) |
| نشر نسخة 0.10.0 مستقرة | **لم يحدث، ولا يجوز استنتاجه من نجاح CI** |

**التفسير العلمي:** ربط اسم دالة بكل بند ليس اعتمادًا علميًا؛ بعض الوظائف تدعم حالات محدودة فقط. يجب مراجعة التصميم والإحصاء والمراجع والاستقلالية قبل استخدامها في دراسة منشورة.

## 1. إعداد نسخة Windows والتحقق منها

في PowerShell داخل جذر المستودع الذي يحتوي `pyproject.toml`:

```powershell
git checkout main
git pull origin main
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,econometrics,plots,exports,unitroot,quantiles]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe tools\release_gate.py --target source
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m twine check dist\*
.\.venv\Scripts\python.exe tools\check_distributions.py dist
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps ".\dist\khaled_dsml-0.10.0rc1-py3-none-any.whl"
.\.venv\Scripts\python.exe -c "import khaled_dsml; print(khaled_dsml.__version__)"
```

الأمر الأخير يعرض `0.10.0rc1` بعد دمج نسخة التحضير. هذه **عملية بناء وتثبيت محلية**، وليست نشرًا على PyPI.

عند الحاجة للميزات الموزعة ثبت `.[dask,spark]` منفصلًا وتحقق من توفر Java اللازمة لـPySpark. الخيار `.[all]` يتضمن المحركات الموزعة وقد يكون ثقيلًا ولا يصلح لبيئات قليلة الموارد.

## 2. التحقق من GitHub Actions

يجب نجاح **سبع مهام** `.github/workflows/python-tests.yml`: أربع مصفوفات Python (3.10، 3.11، 3.12، 3.13)، اختبار Dask/PySpark الحقيقي، **تنفيذ Colab 19–23 فعليًا دون شبكة مع التحقق من ملفات DOCX وXLSX**، ثم بناء ملفات `wheel` و`sdist` والتحقق منهما.

التحقق الجديد `tools/release_gate.py` يطابق نسختي `pyproject.toml` و`khaled_dsml.__version__`، ويتأكد من البنود الـ256 وتوثيق القيود في #85 و#117. وفحص `tools/check_distributions.py` يتحقق من وجود سجل الكتالوج الكامل **داخل أرشيف المصدر**، ومن توافق إصدارات الملفات الناتجة.

## 3. تجربة TestPyPI — تتطلب تشغيلًا يدويًا واضحًا

لا يبدأ النشر إلى TestPyPI تلقائيًا. عند اختيار اختباره:
1. اربط الحساب في https://test.pypi.org/ بالناشر الموثوق `noble2013/KhaledDSML` وملف `testpypi.yml` وبيئة GitHub `testpypi`.
2. افتح GitHub → Actions → **Publish to TestPyPI (manual)** → **Run workflow**.
3. أدخل كلمة الموافقة `TESTPYPI` في `confirm_publish`.
4. ترفض بوابة الاختبار الإصدار المستقر، وتسمح فقط بالنسخ التطويرية/التجريبية؛ تقنيًا يعتمد النشر على إعداد OIDC والبيئة الصحيحين.

بعد نجاح الاختبار، يمكن تثبيت نسخة تجريبية في بيئة **جديدة** بتحميل الاعتماديات الأساسية من PyPI الرسمي أولًا، ثم الحزمة التجريبية بـ`--no-deps`:

```powershell
py -3.13 -m venv .testvenv
.\.testvenv\Scripts\python.exe -m pip install --upgrade pip
.\.testvenv\Scripts\python.exe -m pip install numpy pandas scipy scikit-learn
.\.testvenv\Scripts\python.exe -m pip install --no-deps --index-url https://test.pypi.org/simple/ "khaled-dsml==0.10.0rc1"
.\.testvenv\Scripts\python.exe -c "import khaled_dsml; print(khaled_dsml.__version__)"
```

**تنبيه:** الأمر يفترض أن الإصدار التجريبي نُشر فعلًا؛ لا تنفذه بهدف تنزيل إصدار لم يُنشر بعد.

## 4. نشر PyPI المستقر — إجراء منفصل لاحق

الملف `.github/workflows/release.yml` يُنفذ عند نشر GitHub Release. بوابة الإصدار ترفض تلقائيًا:
- الإصدار الذي يتضمن `dev` أو `a` أو `b` أو `rc`.
- الوسم الذي لا يطابق نسخة `pyproject.toml` تمامًا.
- GitHub Release معلّمًا بأنه `prerelease`.

عندما تقرر نشر `0.10.0` النهائي بعد مراجعة الاختبارات والاستقرار والتوثيق:
1. غيّر **في commit واحد** نسختي `pyproject.toml` و`src/khaled_dsml/__init__.py` إلى `0.10.0`، وعدّل تصنيف الحالة وتوثيق الإصدار.
2. شغّل الاختبارات وبناء وتثبيت الملفات، وراجع الأرقام والتحفظات العلمية.
3. أنشئ الوسم `v0.10.0` من commit مُجاز، ثم GitHub Release بنفس الوسم وغير معلّم `prerelease`.
4. تأكد من حماية بيئة `pypi` في GitHub (مراجعين وأذونات مناسبة) ومن إعداد Trusted Publisher لنفس `release.yml`. النشر بواسطة OIDC لا يحتاج حفظ رموز دخول داخل المشروع.
5. تحقق من الإصدار المنشور على https://pypi.org/project/khaled-dsml/ باختبار تثبيت في بيئة نظيفة.

**لا تنشر PR تجريبيًا أو وسم rc1 كإصدار مستقر.** كما لا يمكن استبدال ملفات PyPI بعد النشر بالاسم والإصدار نفسيهما؛ أخطاء الإصدار تتطلب إصدارًا جديدًا.

## 5. المراجع الرسمية

- [Python Packaging — PEP 440](https://peps.python.org/pep-0440/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [PyPI GitHub OIDC](https://docs.pypi.org/trusted-publishers/)
- [تفاصيل القيود العلمية للـ256 بندًا](CATALOG_AUDIT_0_10_AR_EN.md)
- [بدائل القدرة اللاحقة وفائض الأصفار](ZERO_INFLATION_AND_PRECISION_0_10_AR_EN.md)
