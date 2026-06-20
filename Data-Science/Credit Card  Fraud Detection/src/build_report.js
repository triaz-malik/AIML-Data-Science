// Generates the Credit Card Fraud Detection portfolio report (.docx) with charts.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageBreak, Footer, PageNumber, VerticalAlign,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "reports", "figures");
const OUT = path.join(ROOT, "reports", "Credit_Card_Fraud_Detection_Report.docx");

const BLUE = "1F4E79", RED = "C44E52", GREY = "595959", HEADBG = "1F4E79";

// ---- helpers -------------------------------------------------------------
function img(file, width) {
  const sizes = {
    "01_class_imbalance.png": 0.800, "02_amount_by_class.png": 0.571,
    "03_fraud_by_hour.png": 0.444, "04_fraud_by_category.png": 0.625,
    "07_model_comparison_pr.png": 0.733, "08_pr_auc_bars.png": 0.615,
    "09_shap_beeswarm.png": 0.962, "10_shap_waterfall.png": 0.928,
    "06_baseline_coefficients.png": 1.0,
  };
  const ratio = sizes[file] || 0.6;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(path.join(FIG, file)),
      transformation: { width, height: Math.round(width * ratio) },
      altText: { title: file, description: file, name: file },
    })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text, italics: true, size: 18, color: GREY })],
  });
}
function h1(text) { return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] }); }
function h2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] }); }
function p(runs, opts = {}) {
  const children = Array.isArray(runs) ? runs : [new TextRun(runs)];
  return new Paragraph({ spacing: { after: 120 }, ...opts, children });
}
function bullet(text) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 },
    children: typeof text === "string" ? [new TextRun(text)] : text });
}
const border = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border,
  insideHorizontal: border, insideVertical: border };
function cell(text, w, { bold = false, fill = null, color = null, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align,
      children: [new TextRun({ text, bold, color: color || (fill === HEADBG ? "FFFFFF" : "000000"), size: 19 })] })],
  });
}
function headerRow(labels, widths) {
  return new TableRow({ tableHeader: true,
    children: labels.map((l, i) => cell(l, widths[i], { bold: true, fill: HEADBG, align: AlignmentType.CENTER })) });
}
function row(cells, widths, opts = []) {
  return new TableRow({ children: cells.map((c, i) => cell(c, widths[i], { align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, ...(opts[i] || {}) })) });
}

// ---- document ------------------------------------------------------------
const CW = 9360; // content width (US Letter, 1" margins)

const doc = new Document({
  creator: "Muhammad Tahir Riaz",
  title: "Credit Card Fraud Detection",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: BLUE, font: "Calibri" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "2E5496", font: "Calibri" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Credit Card Fraud Detection  •  Muhammad Tahir Riaz  •  Page ", size: 16, color: GREY }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GREY })] })] }) },
    children: [
      // ---------- Title block ----------
      new Paragraph({ spacing: { before: 1600, after: 0 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Credit Card Fraud Detection", bold: true, size: 56, color: BLUE })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 0 },
        children: [new TextRun({ text: "Detecting fraudulent transactions on a highly imbalanced dataset", size: 26, color: GREY, italics: true })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 0 },
        children: [new TextRun({ text: "An end-to-end machine-learning case study", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 0 },
        children: [new TextRun({ text: "EDA  →  Feature Engineering  →  Model Comparison  →  Tuning  →  SHAP  →  Business Impact", size: 18, color: GREY })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 900, after: 0 },
        children: [new TextRun({ text: "Muhammad Tahir Riaz", bold: true, size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60 },
        children: [new TextRun({ text: "github.com/triaz-malik/Machine-Learning", size: 18, color: BLUE })] }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---------- Executive summary ----------
      h1("Executive Summary"),
      p([
        new TextRun("We built a fraud-detection system on 1.85 million card transactions where only "),
        new TextRun({ text: "0.58% are fraudulent", bold: true }),
        new TextRun(". Because the data is so imbalanced, accuracy is meaningless — a model that predicts “never fraud” scores 99.4% accuracy while catching zero fraud. We therefore optimised "),
        new TextRun({ text: "recall, precision, and PR-AUC", bold: true }),
        new TextRun(" instead."),
      ]),
      p([
        new TextRun("Moving from a linear baseline to a tuned XGBoost model lifted PR-AUC from "),
        new TextRun({ text: "0.18 to 0.88", bold: true, color: RED }),
        new TextRun(". At a business-chosen operating point the model catches "),
        new TextRun({ text: "90% of fraud while flagging only 0.35% of legitimate transactions", bold: true }),
        new TextRun(" for review — turning roughly 5,800 caught frauds (baseline) into ~9,000 per 10,000 fraud cases."),
      ]),

      // ---------- 1. Business problem ----------
      h1("1. Business Problem"),
      p("Banks must decide, in real time, whether to approve or block each card transaction. Fraud is rare but expensive, and the cost structure is asymmetric:"),
      bullet([new TextRun({ text: "Missed fraud (false negative): ", bold: true }), new TextRun("direct financial loss and chargebacks.")]),
      bullet([new TextRun({ text: "False alarm (false positive): ", bold: true }), new TextRun("manual-review cost and customer friction.")]),
      p("The objective is to maximise the share of fraud caught (recall) while keeping false alarms low enough that the review team can cope (precision)."),

      // ---------- 2. Inputs / Data ----------
      h1("2. The Input Data"),
      p([new TextRun({ text: "Dataset: " , bold: true }),
        new TextRun("Simulated credit-card transactions (Sparkov). Pre-split into train and test files.")]),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [3120, 3120, 3120], rows: [
        headerRow(["Split", "Transactions", "Fraud (rate)"], [3120, 3120, 3120]),
        row(["Train", "1,296,675", "7,506  (0.579%)"], [3120, 3120, 3120]),
        row(["Test (held-out)", "555,719", "2,145  (0.386%)"], [3120, 3120, 3120]),
      ] }),
      p("", { spacing: { after: 60 } }),
      img("01_class_imbalance.png", 330),
      caption("Figure 1 — Extreme class imbalance: fraud is 0.58% of all transactions (log scale)."),

      h2("Features used by the model"),
      p("From the raw columns we engineered 23 features. PII and identifiers were dropped to prevent the model memorising individuals, and leakage / non-predictive columns were removed."),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [2600, 6760], rows: [
        headerRow(["Feature group", "Details / rationale"], [2600, 6760]),
        row(["Amount", "amt and log_amt — the single strongest signal."], [2600, 6760]),
        row(["Time of day", "hour and is_night (22:00–03:59), where fraud concentrates."], [2600, 6760]),
        row(["Calendar", "day_of_week, is_weekend."], [2600, 6760]),
        row(["Demographic", "age (from dob), gender, log city population."], [2600, 6760]),
        row(["Merchant category", "14 categories, one-hot encoded."], [2600, 6760]),
        row(["Dropped — PII", "first, last, street, trans_num, raw cc_num."], [2600, 6760]),
        row(["Dropped — leakage", "unix_time (duplicates timestamp)."], [2600, 6760]),
        row(["Dropped — no signal", "home–merchant distance (both classes ~76 km; verified, not assumed)."], [2600, 6760]),
      ] }),

      // ---------- 3. EDA ----------
      h1("3. Exploratory Findings"),
      p("Profiling the full training set surfaced three dominant fraud signals, later confirmed by the model."),
      img("02_amount_by_class.png", 470),
      caption("Figure 2 — Fraud transactions skew to far higher amounts (mean $531 vs $68)."),
      img("03_fraud_by_hour.png", 560),
      caption("Figure 3 — Fraud is a night-time phenomenon: rates spike 22:00–03:59, up to ~25× daytime."),
      img("04_fraud_by_category.png", 440),
      caption("Figure 4 — Online categories (shopping_net, misc_net) carry the highest fraud rates."),

      // ---------- 4. Models ----------
      new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun("4. Models")] }),
      p("We trained an explainable baseline, then progressively stronger models — each evaluated on the same held-out test set with the same metrics. Imbalance was handled with class weights (LogReg, RandomForest) and scale_pos_weight (XGBoost)."),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [2960, 1280, 1280, 1280, 1280, 1280], rows: [
        headerRow(["Model", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"], [2960, 1280, 1280, 1280, 1280, 1280]),
        row(["Logistic Regression (baseline)", "0.184", "0.964", "0.29", "0.58", "0.38"], [2960, 1280, 1280, 1280, 1280, 1280]),
        row(["Random Forest", "0.872", "0.997", "0.90", "0.75", "0.81"], [2960, 1280, 1280, 1280, 1280, 1280]),
        row(["XGBoost", "0.873", "0.998", "0.90", "0.76", "0.83"], [2960, 1280, 1280, 1280, 1280, 1280]),
        row(["XGBoost (tuned)", "0.880", "0.998", "0.90", "0.90", "—"], [2960, 1280, 1280, 1280, 1280, 1280],
          [{ bold: true, fill: "FCE4E4" }, { bold: true, fill: "FCE4E4" }, { fill: "FCE4E4" }, { fill: "FCE4E4" }, { bold: true, fill: "FCE4E4" }, { fill: "FCE4E4" }]),
      ] }),
      caption("Recall for the tuned model is at the business threshold (Section 6); others at best-F1."),
      img("08_pr_auc_bars.png", 430),
      caption("Figure 5 — PR-AUC leaps 0.18 → 0.87 moving from linear to tree models."),
      img("07_model_comparison_pr.png", 430),
      caption("Figure 6 — Precision-recall curves: tree models dominate across the whole range."),
      p([new TextRun({ text: "Why trees win: ", bold: true }),
        new TextRun("fraud lives in interactions — high amount AND night AND an online category. A linear model cannot represent these; gradient-boosted trees can.")]),
      p([new TextRun({ text: "Tuning: ", bold: true }),
        new TextRun("a PR-AUC-scored RandomizedSearchCV selected max_depth=6, learning_rate=0.05, n_estimators=600, subsample=0.7, gamma=1 (CV PR-AUC 0.90).")]),

      // ---------- 5. Explainability ----------
      h1("5. Explainability (SHAP)"),
      p("The model is auditable, not a black box. SHAP confirms the EDA story: amount and night-time dominate, followed by hour and online categories."),
      img("09_shap_beeswarm.png", 430),
      caption("Figure 7 — Global SHAP: log_amt and is_night are the top drivers of fraud probability."),
      img("10_shap_waterfall.png", 450),
      caption("Figure 8 — Local SHAP: why one specific transaction was flagged as fraud."),

      // ---------- 6. Business value ----------
      h1("6. Business Value"),
      p("Bank policy: catch at least 90% of fraud. We pick the highest-precision probability threshold that meets that recall (threshold = 0.80) and read off the operational trade-off on the held-out test set."),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [5360, 4000], rows: [
        headerRow(["Outcome at the 90%-recall threshold", "Result"], [5360, 4000]),
        row(["Frauds caught (true positives)", "1,931 of 2,145  —  90.0%"], [5360, 4000], [{}, { bold: true }]),
        row(["Frauds missed (false negatives)", "214  —  10.0%"], [5360, 4000]),
        row(["False alarms sent to review", "1,910 of 553,574 legit  —  0.35%"], [5360, 4000]),
        row(["Precision (flags that are real fraud)", "50.3%"], [5360, 4000]),
      ] }),
      p("", { spacing: { after: 40 } }),
      p([new TextRun({ text: "Translated to scale — per 10,000 fraud cases:", bold: true })]),
      bullet([new TextRun("Baseline Logistic Regression (58% recall): catches ~5,840, "),
        new TextRun({ text: "misses ~4,160", color: RED })]),
      bullet([new TextRun("Tuned XGBoost (90% recall): catches ~9,000, "),
        new TextRun({ text: "misses ~1,000", color: RED })]),
      p([new TextRun({ text: "Net impact: ", bold: true }),
        new TextRun("missed fraud falls by ~75%, while only 0.35% of legitimate customers experience a review — a strong recall gain at a manageable false-alarm cost.")]),

      // ---------- 7. Conclusion ----------
      h1("7. Conclusion & Reproducibility"),
      p("The project demonstrates the full lifecycle: grounded EDA, leak-safe feature engineering, an iterative model ladder evaluated with the right metrics, hyperparameter tuning, SHAP explainability, and a business-driven operating point."),
      bullet([new TextRun({ text: "Notebooks: ", bold: true }), new TextRun("01 EDA • 02 features + baseline • 03 model comparison • 04 tuning + SHAP + threshold.")]),
      bullet([new TextRun({ text: "Reusable code: ", bold: true }), new TextRun("src/features.py (leak-safe feature pipeline).")]),
      bullet([new TextRun({ text: "Repository: ", bold: true }), new TextRun({ text: "github.com/triaz-malik/Machine-Learning", color: BLUE })]),
      p([new TextRun({ text: "Note: ", bold: true, color: GREY }),
        new TextRun({ text: "the raw CSVs (~500 MB) are not committed; download from the Kaggle “Credit Card Transactions Fraud Detection” dataset to reproduce.", color: GREY, size: 18 })]),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length, "bytes"); });
