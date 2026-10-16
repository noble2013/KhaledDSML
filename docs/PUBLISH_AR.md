# دليل نشر KhaledDSML — GitHub وPyPI

**المؤلف:** الأستاذ الدكتور خالد علي (Khaled Ali)  
**الجامعة:** جامعة الشهيد حمه لخضر – الوادي، الجزائر (University of El Oued, Algeria)  
**الكلية:** كلية العلوم الاقتصادية والتجارية وعلوم التسيير (Faculty of Economic, Commercial and Management Sciences)  
**البريد الإلكتروني المهني:** [ali-khaled@univ-eloued.dz](mailto:ali-khaled@univ-eloued.dz)  
**اسم التوزيع المقترح:** `khaled-dsml`  
**اسم الاستيراد في Python:** `khaled_dsml`

> هذه الملفات تجهّز المشروع للنشر ولا تنشره تلقائيًا. لم يُتحقَّق نهائيًا من توفر الاسم على PyPI. يجب عدم وضع كلمة المرور أو رموز الوصول داخل الملفات أو محادثة أو GitHub.

## أولا: إنشاء مستودع GitHub

1. سجل الدخول في https://github.com/new.
2. أنشئ مستودعًا باسم `KhaledDSML` ضمن حسابك، واجعله **Public** عندما تريد إتاحة المصدر للآخرين. لا تختر إنشاء README أو ترخيص من واجهة GitHub؛ فهما موجودان في المشروع بالفعل.
3. نزّل الحزمة المصدرية وافتح PowerShell من جذر المشروع الذي يحتوي `pyproject.toml`.
4. نفّذ الأوامر التالية **بعد استبدال** `YOUR_GITHUB_USERNAME` باسم مستخدم GitHub الحقيقي (لا تستخدم الاسم النموذجي كما هو):

```powershell
git init
git add .
git commit -m "Initial release of KhaledDSML 0.1.0"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/KhaledDSML.git
git push -u origin main
```

يمكن استخدام تطبيق GitHub Desktop بدل `git` إن لم يكن مثبتًا، ولا حاجة لنشر الحزمة في دليل نظام Windows أو داخل `.venv`.

## ثانيا: التحقق محليًا قبل النشر

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,econometrics,plots]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m twine check dist\*
```

إن لم تكن أدوات البناء متاحة، تثبت ضمن خيار `[dev]` الموضح أعلاه. ارفع الملفات المصدرية **دون** `.venv` و`dist` والملفات المؤقتة.

## ثالثا: إنشاء الحسابات

- **PyPI**: https://pypi.org/account/register/ — حساب نشر الحزم النهائي.
- **TestPyPI**: https://test.pypi.org/account/register/ — حساب **منفصل** للتجارب.

## رابعا: النشر الآمن دون تخزين API Token باستخدام Trusted Publishing

المشروع يحتوي مساري GitHub Actions منفصلين:

| الوجهة | اسم الملف على GitHub | وقت التنفيذ | GitHub Environment |
|---|---|---|---|
| TestPyPI | `.github/workflows/testpypi.yml` | تشغيل يدوي من Actions | `testpypi` |
| PyPI | `.github/workflows/release.yml` | إصدار GitHub Release منشور | `pypi` |

1. من المستودع في GitHub، افتح `Settings → Environments` وأنشئ بيئتين باسم `testpypi` و`pypi`. يوصى بضبط **الموافقة اليدوية** للمراجعين على بيئة `pypi` وقيود حمايتها قبل أي نشر نهائي.
2. في حساب **TestPyPI** افتح https://test.pypi.org/manage/account/publishing/، وأضف **pending Trusted Publisher** بالحقول: اسم الحزمة `khaled-dsml`، مالك مستودع GitHub الفعلي، المستودع `KhaledDSML`، اسم Workflow `testpypi.yml`، وEnvironment `testpypi`.
3. في GitHub ادخل `Actions → Publish to TestPyPI (manual) → Run workflow` على الفرع `main`.
4. بعد اكتمال المهمة ابحث عن الحزمة على https://test.pypi.org/project/khaled-dsml/، وجرّب التثبيت في بيئة فارغة. ثبّت الاعتمادات أولًا من PyPI الرسمي، ثم ثبّت مكتبتنا وحدها من TestPyPI باستعمال `--no-deps` لتفادي خلط المستودعات:

```powershell
py -3.13 -m venv .testvenv
.\.testvenv\Scripts\python.exe -m pip install numpy pandas scipy scikit-learn
.\.testvenv\Scripts\python.exe -m pip install --no-deps --index-url https://test.pypi.org/simple/ khaled-dsml==0.1.0
.\.testvenv\Scripts\python.exe -c "import khaled_dsml; print('Imported OK')"
```

إذا رغبت في تثبيت ميزات إضافية، ثبّت اعتمادياتها أولًا من PyPI الرسمي بحسب خيارات `[econometrics]` و`[plots]` في `pyproject.toml`.

5. بعد نجاح الاختبار، افتح https://pypi.org/manage/account/publishing/ وأضف **pending Trusted Publisher** لاسم المشروع `khaled-dsml`، مالك ومستودع GitHub نفسيهما، وWorkflow `release.yml`، وEnvironment `pypi`.
6. من GitHub أنشئ Release جديدًا، وسمه `v0.1.0`، ثم اضغط `Publish release`. يتحقق workflow من اختبارات المشروع وتطابق وسم الإصدار مع `pyproject.toml`، ويُنشئ الحزم، ثم ينشرها إلى PyPI تلقائيًا **فقط بعد** إتمام Trusted Publisher والصلاحيات المطلوبة.
7. تأكد من ظهور https://pypi.org/project/khaled-dsml/ ثم نفذ الأمر التالي لاختبار التثبيت العام:

```powershell
py -3.13 -m pip install khaled-dsml
```

## خامسا: لماذا بيئتان مختلفتان؟

يتحقق PyPI من هوية GitHub بواسطة OIDC قصيرة الأجل دون تخزين أسرار نشر ثابتة، وبالاعتماد على اسم المستودع وملف العمل واسم البيئة. لهذا يجب كتابة أسماء الملفات والبيئات **كما هي حرفيًا**. المستودع وحده لا يكفي للنشر، والحسابان على PyPI وTestPyPI مستقلان.

## سادسا: ما الذي ينشره الإصدار 0.1.0؟

- وحدات إحصائية وتعلم آلة وجودة بيانات، واختبارات آلية مع تفسير عربي.
- وظائف لمعالجة الملفات بالأجزاء وأمثلة اقتصاد كمي.
- أمثلة Jupyter وملفات توثيق عربية.
- **لا** يشمل تنفيذًا متكاملاً لجميع وظائف Spark أو Hadoop أو NoSQL، ولا يثبت تغطيته لكل محتوى بطاقة دراسية رسمية لم تتم مراجعتها.

## مصادر التوثيق الرسمية

- https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- https://docs.pypi.org/trusted-publishers/using-a-publisher/
- https://docs.pypi.org/trusted-publishers/security-model/
- https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
