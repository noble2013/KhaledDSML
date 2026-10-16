"""Optional classroom multiple-choice quizzes with reproducible selection."""
from __future__ import annotations
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Question:
    id: str
    course: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str
    explanation_ar: str


BANK = [
    Question("DS01", "data_science", "ما المقصود بوحدة المشاهدة؟", ("سطر يمثل حالة", "اسم الجدول", "نوع الرسم", "ملف البرنامج"), "A", "المشاهدة هي الحالة التي يمثلها صف الجدول."),
    Question("DS02", "data_science", "أي مقياس أقل تأثرًا بالقيم المتطرفة؟", ("المتوسط", "الوسيط", "المدى", "التباين"), "B", "الوسيط مقاوم نسبيًا للقيم المتطرفة."),
    Question("DS03", "data_science", "ما المشكلة الأساسية في حذف جميع القيم المفقودة؟", ("زيادة حجم البيانات", "فقد عينة واحتمال التحيز", "تحسين التمثيل دائمًا", "تقليل الدقة العددية فقط"), "B", "الحذف قد يقلص العينة ويسبب تحيزًا."),
    Question("DS04", "data_science", "أي اختبار شائع لفحص اعتدالية توزيع البيانات؟", ("Shapiro-Wilk", "K-means", "F1", "MAE"), "A", "Shapiro-Wilk يختبر فرضية التوزيع الطبيعي."),
    Question("DS05", "data_science", "ماذا يعني الارتباط الموجب؟", ("السببية مؤكدة", "اتجاه التغير المشترك غالبًا واحد", "عدم وجود علاقة", "تساوي المتوسطات"), "B", "الارتباط لا يثبت السببية."),
    Question("DS06", "data_science", "ماذا يعني عدم رفض H0؟", ("أثبتنا صحة H0", "لا توجد أدلة كافية للرفض", "H1 صحيحة حتمًا", "حجم العينة كبير"), "B", "عدم الرفض لا يثبت الفرضية الصفرية."),
    Question("DS07", "data_science", "الهدف من IQR في التنظيف؟", ("قياس ارتباط خطي", "كشف قيم بعيدة وفق الربيعات", "ترتيب الأشهر", "تقسيم التدريب"), "B", "قاعدة IQR إحدى طرق الإشارة إلى الشواذ."),
    Question("DS08", "data_science", "ماذا تتطلب معالجة البيانات الأخلاقية؟", ("نشر بيانات العملاء", "حماية الخصوصية وتوثيق المصدر", "حذف مصدر البيانات", "إلغاء النسخ الاحتياطية"), "B", "إخفاء البيانات الحساسة وأخذ الموافقات من الأسس."),
    Question("ML01", "machine_learning", "ما دور مجموعة الاختبار؟", ("ضبط كل المعاملات حتى أفضل نتيجة", "قياس التعميم النهائي", "تدريب النموذج", "حذف الفئات"), "B", "مجموعة الاختبار تعطي تقديرًا على بيانات غير مستخدمة بالتدريب/الضبط."),
    Question("ML02", "machine_learning", "ما مثال على تسرب البيانات؟", ("التقييس قبل تقسيم كامل البيانات", "استخدام Pipeline داخل CV", "فصل البيانات أولًا", "تثبيت random_state"), "A", "التقييس قبل التقسيم ينقل معلومات الاختبار إلى التدريب."),
    Question("ML03", "machine_learning", "أي مقياس يستخدم للتنبؤ العددي؟", ("Recall", "MAE", "Accuracy", "F1"), "B", "MAE متوسط الخطأ المطلق."),
    Question("ML04", "machine_learning", "ما وظيفة Ridge؟", ("عقوبة L2", "تجميع غير خاضع للإشراف", "تشخيص بيانات نصية", "تحويل JSON"), "A", "Ridge يطبق تنظيم L2 في الانحدار."),
    Question("ML05", "machine_learning", "أي خوارزمية تستخدم لتجميع عملاء دون تسميات؟", ("K-means", "LogisticRegression", "LinearRegression", "Welch t-test"), "A", "K-means تعلم غير خاضع للإشراف."),
    Question("ML06", "machine_learning", "ما سبب استخدام F1 مع الفئات غير المتوازنة؟", ("يلغي تحيز البيانات", "يوفق بين Precision وRecall", "يقيس المبيعات", "يلغي الحاجة إلى الاختبار"), "B", "F1 متوسط توافقي للدقة الإيجابية والاسترجاع."),
    Question("ML07", "machine_learning", "كيف تختبر عدة تقسيمات للتدريب والتقييم؟", ("Cross-validation", "Excel فقط", "إزالة المتغير الهدف", "تقريب الأرقام"), "A", "التحقق المتقاطع يعيد التقييم عبر طيات متعددة."),
    Question("ML08", "machine_learning", "ما المقصود بـ PCA؟", ("اختبار قيمة مفقودة", "خفض الأبعاد بمركبات رئيسية", "تصنيف موجه", "توليد SQL"), "B", "PCA يستخرج مركبات تفسر تباينًا من البيانات."),
    Question("BD01", "big_data", "ما المقصود بـ Volume في 5Vs؟", ("حجم البيانات", "دقتها", "مصدرها", "قيمتها الاقتصادية"), "A", "Volume يشير إلى حجم البيانات."),
    Question("BD02", "big_data", "ماذا تفيد chunksize في read_csv؟", ("تحميل الملف كله دائمًا", "قراءة دفعات بدل الملف كاملًا", "إزالة الارتباط", "ترميز متغير الهدف"), "B", "chunksize يعيد أجزاء متعاقبة للمعالجة."),
    Question("BD03", "big_data", "ما المقصود بـ MapReduce؟", ("رسم النقاط", "توزيع المعالجة ثم تجميع النتائج", "إعادة تسمية الأعمدة", "فحص الاعتدالية"), "B", "Map ثم Reduce من مفاهيم المعالجة الموزعة."),
    Question("BD04", "big_data", "أي نوع تمثله مستندات JSON غالبًا؟", ("شبه منظمة", "صور خام", "غير قابلة للمعالجة", "بيانات مقاسة بالثواني"), "A", "JSON يستخدم بنية مفاتيح وقيم مرنة."),
    Question("BD05", "big_data", "كيف نقارن استهلاك الذاكرة لـ DataFrame؟", ("memory_usage(deep=True)", "shape وحده", "head", "unique"), "A", "deep=True يحسب أيضًا الذاكرة المخزنة في كائنات النصوص."),
    Question("BD06", "big_data", "ما الفرق بين SQL وNoSQL؟", ("NoSQL دائمًا أسرع", "NoSQL تدعم نماذج تخزين متعددة", "SQL لا تدعم الفهارس", "SQL تعني الصور"), "B", "NoSQL تشمل المستندات والمفاتيح/القيم وأشكالًا أخرى."),
    Question("BD07", "big_data", "ما خطر استخلاص استنتاجات عامة من بيانات اصطناعية؟", ("لا توجد مخاطر", "قد لا تمثل الواقع الاقتصادي", "تزيد السرعة", "تزيد عدد الأعمدة"), "B", "بيانات المحاكاة توضح المنهج لكنها ليست دليلًا واقعيًا."),
    Question("BD08", "big_data", "ما الاسم الشائع للتنبؤ الآني بمؤشر اقتصادي؟", ("Nowcasting", "OneHot", "Shapiro", "KNN"), "A", "Nowcasting تقدير للحالة الحالية قبل إصدار الإحصاءات النهائية."),
]


def get_questions(course="all", limit=10, seed=42) -> list[Question]:
    courses = ("all", "data_science", "machine_learning", "big_data")
    if course not in courses:
        raise ValueError("course must be: " + ", ".join(courses))
    if limit < 1:
        raise ValueError("limit must be positive")
    questions = [q for q in BANK if course == "all" or q.course == course]
    return Random(seed).sample(questions, min(limit, len(questions)))


def grade_quiz(questions: list[Question], answers: dict[str, str]) -> dict:
    """Grade a list of questions. Answers map question IDs to A/B/C/D."""
    results = [{"id": q.id, "correct": answers.get(q.id, "").strip().upper() == q.answer,
                "answer": q.answer, "explanation_ar": q.explanation_ar} for q in questions]
    count = sum(r["correct"] for r in results)
    return {"correct": count, "total": len(results),
            "percentage": round(count / len(results) * 100, 2) if results else 0,
            "details": results}
