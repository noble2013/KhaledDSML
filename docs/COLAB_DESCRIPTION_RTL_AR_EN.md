# KhaledDSML — بطاقة المؤلف وعرض الوصف في Colab | Author and RTL description

## الصيغة المعتمدة | Current source identity

| الحقل | المعلومة |
|---|---|
| الاسم | خالد |
| اللقب | علي |
| الاسم بالإنجليزية | Khaled Ali |
| اللقب الأكاديمي السابق في ملف README | الأستاذ الدكتور (Professor / Dr.) |
| الجامعة | جامعة الشهيد حمه لخضر – الوادي، الجزائر |
| الكلية | كلية العلوم الاقتصادية والتجارية وعلوم التسيير |
| البريد الإلكتروني المهني | ali-khaled@univ-eloued.dz |

**تنبيه الدقة:** «الأستاذ الدكتور» هو **اللقب المكتوب في README سابقًا**، ولا يكفي وحده لإثبات الرتبة الوظيفية الرسمية (مثل أستاذ التعليم العالي أو أستاذ محاضر قسم «أ»). لا نضيف رتبة إدارية محددة إلا بعد تأكيدها من المؤلف.

## العرض الصحيح في Google Colab

افتح [Colab](https://colab.research.google.com/) ثم استخدم الدفتر الجاهز
[24_library_description_rtl_author.ipynb](../notebooks/24_library_description_rtl_author.ipynb).

الخلية الأولى تثبت `markdown` و`beautifulsoup4` و`requests`.
الخلية الثانية:
1. تجلب أحدث `README.md` **من `main` على GitHub**، وليس وصفًا مخزَّنًا في إصدار قديم على PyPI.
2. تحوّل Markdown إلى HTML مع الاحتفاظ بالجداول والمقاطع البرمجية والروابط.
3. تكتشف المقاطع العربية وتُظهرها باتجاه `dir="rtl"` ومحاذاة يمينية، والإنجليزية باتجاه `ltr` ومحاذاة يسارية.
4. تجعل أمثلة Python وTerminal باتجاه LTR مستقلةً عن لغة النص.
5. تعرض بطاقة المؤلف الموجودة في بداية README بأسماء الحقول العربية.

### ملاحظة حول تحديث PyPI | Important PyPI distinction

نجح نشر `0.10.0` على [PyPI](https://pypi.org/project/khaled-dsml/0.10.0/) بتاريخ 17 أكتوبر 2026.
التعديل في `README.md` على GitHub يُعرض فور دمجه في `main`، لكن **لا يُحدِّث نسخة README المضمنة في ملف التوزيع المنشور مسبقًا**. لإظهار الوصف المعدل داخل صفحة PyPI نفسها يلزم **إصدار أحدث مستقل** يتضمن الوصف الجديد؛ لا يمكن استبدال نفس الملف المنشور `0.10.0` دون تغيير رقم النسخة.

## English

The author card on GitHub deliberately separates `first name = Khaled` and `surname = Ali`, institution, faculty and pre-existing academic title. The university's formal employment rank was **not independently confirmed** and must not be guessed. The Colab viewer uses right-to-left layout for Arabic, left-to-right for English, and left-to-right for code blocks. It retrieves the live GitHub README and requires network access.
