# 📤 دليل رفع المشروع على GitHub — خطوة بخطوة

> مشروع SEC619 — Reem Fuad Shareef

---

## 📋 المتطلبات قبل البدء

- حساب على [github.com](https://github.com) (مجاني)
- تثبيت [Git](https://git-scm.com/downloads) على جهازك
- تثبيت [GitHub Desktop](https://desktop.github.com/) (اختياري — أسهل للمبتدئين)

---

## 🛤️ الطريقة الأولى: عبر GitHub Desktop (الأسهل)

### الخطوة 1 — تحميل وتثبيت GitHub Desktop
اذهبي إلى: https://desktop.github.com وحمّلي البرنامج وثبّتيه.

### الخطوة 2 — سجّلي الدخول
افتحي GitHub Desktop → Sign in to GitHub.com → سجّلي بحسابك.

### الخطوة 3 — أنشئي Repository جديد على github.com
1. اذهبي إلى https://github.com/new
2. املئي:
   - **Repository name:** `SEC619-Threat-Detection`
   - **Description:** `LLM-Driven Digital Threat Detection in Spoken Communication — KFUPM Graduation Project`
   - **Visibility:** Public أو Private (حسب رغبتك)
   - ❌ لا تضغطي على "Add a README file" (لأن عندك واحد جاهز)
3. اضغطي **Create repository**

### الخطوة 4 — ربط المجلد بالريبو
في GitHub Desktop:
1. File → Add Local Repository
2. اختاري مجلد مشروعك
3. سيطلب منك Publish — اضغطي **Publish repository**

### الخطوة 5 — رفع الملفات (Commit & Push)
1. ستظهر كل ملفاتك في قائمة Changes
2. اكتبي في خانة **Summary:** `Initial commit: SEC619 graduation project`
3. اضغطي **Commit to main**
4. اضغطي **Push origin**

✅ **تم! مشروعك الآن على GitHub.**

---

## 🖥️ الطريقة الثانية: عبر Command Line (Git)

### الخطوة 1 — تحقّقي من تثبيت Git
```bash
git --version
```
إذا ظهر رقم إصدار فالأمر تمام. إن لم يكن مثبتاً، حمّليه من https://git-scm.com/downloads

### الخطوة 2 — أنشئي Repository على github.com
1. اذهبي إلى https://github.com/new
2. اسم الريبو: `SEC619-Threat-Detection`
3. اتركي الخيارات كما هي (لا تضيفي README)
4. اضغطي **Create repository**
5. انسخي الرابط مثل: `https://github.com/YOUR_USERNAME/SEC619-Threat-Detection.git`

### الخطوة 3 — جهّزي مجلد المشروع
افتحي Terminal أو Command Prompt وانتقلي لمجلد المشروع:

```bash
cd "C:\Users\Reem\Grad Project"
```

### الخطوة 4 — تهيئة Git
```bash
git init
git branch -M main
```

### الخطوة 5 — ربط الريبو البعيد
```bash
git remote add origin https://github.com/YOUR_USERNAME/SEC619-Threat-Detection.git
```
> استبدلي `YOUR_USERNAME` باسم المستخدم عندك على GitHub

### الخطوة 6 — إضافة الملفات والـ Commit
```bash
git add README.md
git add requirements.txt
git add .gitignore
git add GITHUB_UPLOAD_GUIDE.md
git add "100_Dataset/build_dataset_100.py"
git add "100_Dataset/tts_edge_emotional_large.py"
git add "100_Dataset/dataset_100_samples.json"
git add "100_Dataset/GUIDE.md"
git add "100_Dataset/Output/"
```

> ⚠️ **ملاحظة مهمة:** ملفات الصوت (.wav) كبيرة الحجم. إذا حجمها كبير (أكثر من 100MB إجمالاً)، لا ترفعيها عبر Git العادي. اطّلعي على قسم **رفع الملفات الكبيرة** أدناه.

```bash
git commit -m "Initial commit: SEC619 LLM-Driven Threat Detection graduation project

- Dataset builder script (100 samples: 50 Unsafe / 50 Safe)
- Edge TTS emotional audio generator
- Dataset manifest (JSON + XLSX)
- Evaluation output reports (CSV + TXT)
- Full requirements.txt and .gitignore"
```

### الخطوة 7 — رفع المشروع (Push)
```bash
git push -u origin main
```
> سيطلب منك إدخال اسم المستخدم وكلمة المرور (أو Personal Access Token).

---

## 🔐 إنشاء Personal Access Token (PAT)

إذا طلب منك GitHub كلمة مرور عند الـ push:

1. اذهبي إلى: https://github.com/settings/tokens
2. اضغطي **Generate new token (classic)**
3. اكتبي اسماً له مثل: `SEC619-Project`
4. حددي صلاحية: ✅ **repo**
5. اضغطي **Generate token**
6. **انسخي الـ Token** — لن يظهر مرة أخرى!
7. استخدميه بدلاً من كلمة المرور عند الـ push

---

## 📦 رفع ملفات الصوت الكبيرة (.wav)

ملفات الـ WAV قد تكون كبيرة جداً. الحلول:

### الخيار الأول: Git Large File Storage (LFS)
```bash
# تثبيت Git LFS
git lfs install

# تتبع ملفات WAV
git lfs track "*.wav"

# إضافة ملف .gitattributes
git add .gitattributes
git commit -m "Track WAV files with Git LFS"

# رفع كالمعتاد
git add "100_Dataset/Audio/"
git commit -m "Add 100 audio samples"
git push
```
> ⚠️ GitHub يعطي 1GB مجاناً لـ LFS. إذا البيانات أكبر، استخدمي الخيار الثاني.

### الخيار الثاني: Hugging Face Datasets (موصى به للمشاريع الأكاديمية)
1. أنشئي حساب على https://huggingface.co
2. أنشئي Dataset جديد واحتفظي بالرابط
3. أضيفي في README:
```markdown
## Dataset
Audio files are hosted on Hugging Face:
🤗 [SEC619 Dataset](https://huggingface.co/datasets/YOUR_USERNAME/SEC619-dataset)
```

---

## ✅ قائمة تحقق قبل الرفع

- [ ] README.md موجود ومكتوب باحترافية
- [ ] requirements.txt يحتوي على كل المكتبات
- [ ] .gitignore يستثني الملفات غير الضرورية
- [ ] لا توجد بيانات حساسة (API keys, passwords)
- [ ] الكود منظّم في مجلدات واضحة
- [ ] أسماء الملفات بالإنجليزية (لا مسافات أو أحرف عربية)

---

## 🎯 نصائح لريبو احترافي

1. **أضيفي Topics** في صفحة الريبو: `machine-learning`, `nlp`, `speech-recognition`, `content-moderation`, `python`
2. **فعّلي GitHub Pages** لعرض GUIDE.md كموقع ويب (اختياري)
3. **أضيفي License** إذا أردتِ: `Academic Free License`
4. **أضيفي About** في الـ sidebar مع وصف قصير

---

## 🔗 الرابط النهائي للمشروع

بعد الرفع، رابط مشروعك سيكون:
```
https://github.com/YOUR_USERNAME/SEC619-Threat-Detection
```

---

*تم إعداد هذا الدليل لمشروع التخرج SEC619 — Reem Fuad Shareef · KFUPM Term 242*
