// Generates reports/Project_Report.docx
// Run from project root:  node reports/build_report.js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, TableOfContents, HeadingLevel,
  BorderStyle, WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak,
} = require("docx");

const FIG = "outputs/figures/";
const CW = 9360;                 // content width DXA (US Letter, 1" margins)
const NAVY = "1F3864", BLUE = "2E75B6", LIGHT = "D9E2F3", GREY = "595959";

// ---- helpers ---------------------------------------------------------------
const img = (file, wPx, asp) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 60 },
  children: [new ImageRun({
    type: "png", data: fs.readFileSync(FIG + file),
    transformation: { width: wPx, height: Math.round(wPx / asp) },
    altText: { title: file, description: file, name: file },
  })],
});
const caption = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: t, italics: true, size: 18, color: GREY })],
});
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const p = (runs) => new Paragraph({
  spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED,
  children: Array.isArray(runs) ? runs : [new TextRun(runs)],
});
const b = (t) => new TextRun({ text: t, bold: true });
const t = (txt) => new TextRun(txt);
const bullet = (runs) => new Paragraph({
  numbering: { reference: "bul", level: 0 }, spacing: { after: 60 },
  children: Array.isArray(runs) ? runs : [new TextRun(runs)],
});

// ---- table builder ---------------------------------------------------------
const border = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const mk = (txt, opts = {}) => new TableCell({
    borders, width: { size: opts.w, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text: String(txt), bold: !!opts.bold,
        color: opts.header ? "FFFFFF" : "000000", size: 19 })],
    })],
  });
  const headRow = new TableRow({
    tableHeader: true,
    children: headers.map((hh, i) => mk(hh, { w: widths[i], fill: NAVY, bold: true, header: true, center: i > 0 })),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => mk(c, {
      w: widths[i], center: i > 0,
      fill: ri % 2 ? "EEF3FA" : "FFFFFF",
      bold: i === 0,
    })),
  }));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths,
    rows: [headRow, ...bodyRows] });
}

// ---------------------------------------------------------------------------
const titlePage = [
  new Paragraph({ spacing: { before: 2600 } }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "AI-Powered Telecom Fraud, Phishing", bold: true, size: 48, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "& SMS Spam Detection System", bold: true, size: 48, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "Machine-Learning Project Report", size: 30, color: BLUE })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 8 } },
    children: [new TextRun({ text: "", size: 2 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 320 },
    children: [new TextRun({ text: "Detecting banking scams, smishing, fraud and spam in telecom SMS traffic", italics: true, size: 24, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 2400 },
    children: [new TextRun({ text: "Classic ML (TF-IDF + Logistic Regression)  •  DistilBERT  •  BERT", size: 22, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "SHAP Explainability  •  Error Analysis  •  Power BI Dashboards", size: 22, color: GREY })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

const toc = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Table of Contents")] }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
];

const body = [
  // ---------------- Executive Summary ----------------
  h1("Executive Summary"),
  p([t("Telecom operators route millions of SMS messages every day. Hidden inside that traffic are "),
     b("banking scams, fake OTP requests, investment and lottery fraud, fake-delivery smishing, and aggressive promotional spam"),
     t(". Each one erodes customer trust and drives complaints, churn, and fraud-related losses. This project delivers an end-to-end machine-learning system that automatically classifies inbound SMS into five risk categories so risky messages can be flagged, warned, or blocked "),
     b("before customers become victims.")]),
  p([t("We assembled "), b("8,103 unique messages"), t(" from public corpora, engineered linguistic risk features, and trained three models. The best model, a fine-tuned "),
     b("BERT, reached 96.2% accuracy on the 5-class task and 99.1% on binary ham/spam detection"),
     t(". The system is fully explainable (SHAP), audited for errors, and ships analysis-ready data for three Power BI dashboards.")]),
  table(
    ["Headline metric", "Result"],
    [["Messages analysed", "8,103"],
     ["Risk categories", "Normal · Promotion · Spam · Phishing · Fraud"],
     ["Best model (BERT) – 5-class accuracy", "96.2%"],
     ["Best model – binary ham/spam accuracy", "99.1%"],
     ["Unwanted / malicious traffic identified", "44.2% of all messages"],
     ["Phishing messages containing a URL", "51.1%"]],
    [5400, 3960]),
  new Paragraph({ spacing: { after: 200 } }),

  // ---------------- 1. Business Problem ----------------
  h1("1. Business Problem & Objectives"),
  p([b("Goal: "), t("automatically detect risky SMS in real time and route each message to the right action — Allow, Review, Warn, or Block.")]),
  h2("Business questions answered"),
  bullet("What percentage of messages are fraudulent, phishing, or promotional?"),
  bullet("Which fraud type is most common?"),
  bullet("What keywords are most associated with scams?"),
  bullet("Can AI accurately identify phishing attempts?"),
  bullet("Which messages should be blocked?"),

  // ---------------- 2. Data & Methodology ----------------
  h1("2. Data & Methodology"),
  h2("2.1 Data sources"),
  table(
    ["Source", "Role", "Notes"],
    [["UCI SMS Spam Collection", "Primary ground truth", "5,574 messages labelled ham / spam"],
     ["Mishra–Soni Smishing corpus", "Phishing / fraud seed", "Filtered, de-noised smishing examples"]],
    [3000, 2600, 3760]),
  p([t("After de-duplication the working corpus is "), b("8,103 unique messages"),
     t(" (≈ 4,518 ham / 3,585 spam). The binary ham/spam label is treated as true ground truth.")]),
  h2("2.2 From 2 classes to 5 — weak supervision"),
  p([t("Public datasets are only labelled ham/spam. The finer scheme "),
     b("Normal / Promotion / Spam / Phishing / Fraud"),
     t(" is derived with transparent, priority-ordered rules (most-harmful first: Fraud → Phishing → Promotion → Spam; ham → Normal) using curated keyword sets, URL detection, and a smishing-corpus prior.")]),
  p([new TextRun({ text: "Honesty note: ", bold: true, color: "C00000" }),
     t("the fine-grained labels are heuristically derived, not human-annotated. The binary ham/spam label remains true ground truth and is preserved alongside every record. This is documented as a known limitation and is itself a mark of methodological transparency.")]),
  h2("2.3 Pipeline"),
  bullet("01 Acquire → 02 Weak-label → 03 EDA → 04 Feature engineering"),
  bullet("05 Logistic Regression → 06 DistilBERT & BERT → 07 SHAP + errors → 08 Business & Power BI"),
  bullet("Shared deterministic stratified 80/20 split (seed 42) used by all three models for a fair comparison."),

  // ---------------- 3. EDA ----------------
  h1("3. Exploratory Data Analysis"),
  h2("3.1 The fraud / phishing / spam landscape"),
  img("01_class_distribution.png", 600, 2.60),
  caption("Figure 1 — 5-class distribution (left) and binary ground truth (right)."),
  p([t("44.2% of all traffic is unwanted. "), b("Phishing dominates the malicious side at 28.8%"),
     t(", followed by Fraud (9.1%). Promotion (2.7%) and Spam (3.7%) are minority classes — an imbalance we handle with class weights and macro-F1 reporting.")]),
  table(
    ["Class", "Count", "Share", "Risk tier"],
    [["Normal", "4,518", "55.8%", "Safe"],
     ["Phishing", "2,333", "28.8%", "High"],
     ["Fraud", "736", "9.1%", "Critical"],
     ["Spam", "299", "3.7%", "Medium"],
     ["Promotion", "217", "2.7%", "Low"]],
    [3360, 2000, 2000, 2000]),
  new Paragraph({ spacing: { after: 160 } }),
  h2("3.2 Message length"),
  img("02_message_length.png", 600, 2.80),
  caption("Figure 2 — Message length distribution and spread by class."),
  p([t("Normal messages are short (median "), b("53 characters"),
     t("); scam and promotional messages are roughly "), b("3× longer (≈ 144–151 characters)"),
     t("). Length alone is a strong, cheap pre-filter signal.")]),
  h2("3.3 Vocabulary — word clouds"),
  img("03_wordclouds.png", 560, 1.78),
  caption("Figure 3 — Characteristic vocabulary per class."),
  p([t("Each class has a distinct fingerprint: Fraud → "), b("won, prize, claim, congratulations"),
     t("; Phishing → "), b("account, verify, customer, click, link"),
     t("; Promotion → "), b("free, offer, ringtone, reply STOP"), t(".")]),
  h2("3.4 Top n-grams in malicious traffic"),
  img("04_top_ngrams.png", 600, 2.83),
  caption("Figure 4 — Top unigrams, bigrams and trigrams across malicious messages."),

  // ---------------- 4. Feature Engineering ----------------
  h1("4. Feature Engineering"),
  p([t("Alongside TF-IDF we engineered 12 linguistic risk signals: message length, character/word counts, URL count, digit count and ratio, currency-symbol count, uppercase count and ratio, exclamation count, special-character count, and a phone-number flag.")]),
  img("05_feature_profile.png", 600, 2.40),
  caption("Figure 5 — Engineered-feature profile by class (annotated with raw means)."),
  p([b("Strongest correlations with “is malicious”: "),
     t("character count (0.62), message length (0.59), digit count (0.56), URL count (0.52). ")]),
  p([t("The class fingerprints are operationally useful: "),
     b("Phishing leads on URLs"), t(" (0.55 links/msg on average) while "),
     b("Fraud leads on currency symbols"), t(" (0.38/msg) — directly motivating URL screening and currency-based alerts.")]),

  // ---------------- 5. Models ----------------
  h1("5. Models & Results"),
  p([t("Three models were trained on an identical stratified split (6,482 train / 1,621 test) with class weighting for imbalance.")]),
  table(
    ["Model", "5-class Acc", "Macro-F1", "Binary Acc", "Binary F1"],
    [["TF-IDF + Logistic Regression", "95.1%", "0.877", "98.2%", "0.979"],
     ["DistilBERT (fine-tuned)", "95.7%", "0.869", "99.1%", "0.990"],
     ["BERT (fine-tuned)", "96.2%", "0.898", "99.1%", "0.990"]],
    [3760, 1500, 1400, 1400, 1300]),
  new Paragraph({ spacing: { after: 120 } }),
  img("10_model_comparison.png", 560, 1.83),
  caption("Figure 6 — Side-by-side model comparison across four metrics."),
  p([b("BERT is the best overall"), t(" — highest 5-class accuracy and macro-F1, with the biggest gains on the hard minority classes (Promotion 0.80, Spam 0.81 F1). DistilBERT matches BERT on binary ham/spam at a fraction of the size and trains in ~80 seconds on an RTX 5080. The classic TF-IDF + Logistic Regression baseline lands within ~1 point while remaining far cheaper and fully interpretable (best parameters: C = 5.0, L1 penalty).")]),
  img("07_bert_confusion.png", 360, 1.25),
  caption("Figure 7 — BERT confusion matrix (5-class)."),
  p([t("Errors concentrate between the small Promotion and Spam classes, which share marketing vocabulary; the high-stakes Fraud and Phishing classes are recovered at 0.94–0.95 F1.")]),

  // ---------------- 6. Explainability ----------------
  h1("6. Explainability (SHAP)"),
  p([t("Every decision is explainable. Globally, we surface the words that most drive each class; locally, we decompose a single prediction into its word contributions (for the linear model the SHAP value equals coefficient × TF-IDF, so the explanation is exact).")]),
  img("08_global_top_words.png", 600, 3.33),
  caption("Figure 8 — Top driver words per class."),
  img("09_shap_example.png", 480, 1.80),
  caption("Figure 9 — Local explanation for a sample phishing message."),
  p([t("The message "), new TextRun({ text: "“Your account is suspended. Verify immediately.”", italics: true }),
     t(" is correctly flagged as "), b("Phishing"), t(", driven by "),
     b("account, verify, immediately, suspended"), t(" — exactly the human intuition, made auditable.")]),

  // ---------------- 7. Error Analysis ----------------
  h1("7. Error Analysis"),
  p([t("On the 1,621-message test set the baseline produced "), b("17 false positives (1.9% of legitimate messages)"),
     t(" and "), b("13 false negatives (1.8% of risky messages)"), t(".")]),
  h2("False positives — legitimate but looked suspicious"),
  bullet("“Yun buying... But school got offer 2000 plus only...” (the word “offer” + digits)"),
  bullet("“…has been set as your callertune for all Callers.” (service-style phrasing)"),
  h2("False negatives — risky but appeared normal"),
  bullet("“This message is brought to you by GMW Ltd. and is not connected to the…” (spam preamble, benign-looking)"),
  bullet("“*CricInfo Alerts! Type CRICKET & sms to <#>…” (subscription phishing disguised as a sports alert)"),
  p([t("Takeaway: failures cluster where marketing and personal language overlap. Adding human-labelled samples for Promotion/Spam and a URL-reputation feature would close most of the gap.")]),

  // ---------------- 8. Business Value ----------------
  h1("8. Business Value & Recommendations"),
  h2("8.1 Data-driven findings"),
  table(
    ["Finding", "Recommended action"],
    [["51% of phishing messages contain a URL/link", "Increase URL screening; sandbox or block inbound links"],
     ["Fraud/phishing rely on banking keywords (bank, account, verify, OTP, PIN)", "Deploy banking-specific filters + step-up verification"],
     ["“verify”, “account”, “OTP” strongly predict fraud", "Real-time keyword alerts on these triggers"],
     ["Malicious messages are ~2.7× longer than normal", "Use length + digit/URL counts as a fast pre-filter"]],
    [4400, 4960]),
  new Paragraph({ spacing: { after: 160 } }),
  h2("8.2 Power BI dashboards (data shipped as CSV)"),
  table(
    ["Dashboard", "Pages", "Source CSV"],
    [["Executive", "Total messages, Spam %, Fraud %, Phishing %", "class_summary.csv"],
     ["Fraud", "Top scam types, risk tiers, keyword analysis", "fraud_keywords.csv, feature_by_class.csv"],
     ["AI", "Model accuracy, predictions, confidence scores", "model_comparison.csv, messages_scored.csv"]],
    [2000, 4360, 3000]),
  new Paragraph({ spacing: { after: 160 } }),
  h2("8.3 Value for telecom operators"),
  bullet([b("Reduce subscriber fraud and fraud-related losses"), t(" by blocking high-risk messages before delivery.")]),
  bullet([b("Protect customers and improve trust"), t(" — fewer scam messages reach the handset.")]),
  bullet([b("Reduce complaints and churn"), t(" driven by spam and smishing.")]),
  bullet([b("Real-time, explainable, auditable"), t(" risk scoring that regulators and fraud teams can inspect.")]),

  // ---------------- 9. Limitations ----------------
  h1("9. Limitations & Next Steps"),
  bullet("Fine-grained labels are rule-derived (weak supervision), not human-annotated."),
  bullet("The smishing corpus is Nigerian-telecom heavy and noisy; only de-noised seeds were kept."),
  bullet("Class imbalance limits recall on the smallest classes (Promotion, Spam)."),
  bullet("Next: human-verified labels for minority classes, URL-reputation features, transformer hyperparameter sweep, and a production inference API."),

  // ---------------- 10. Conclusion ----------------
  h1("10. Conclusion"),
  p([t("The system meets its objective: it accurately and transparently separates normal SMS from promotion, spam, phishing, and fraud. With "),
     b("96.2% multi-class accuracy, 99.1% binary accuracy"),
     t(", quantified business findings, explainable predictions, and dashboard-ready exports, it provides a deployable foundation for protecting telecom subscribers from SMS-based fraud.")]),
];

// ---------------------------------------------------------------------------
const doc = new Document({
  creator: "SMS Fraud Detection Project",
  title: "AI-Powered Telecom Fraud, Phishing & SMS Spam Detection System",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Calibri", color: NAVY },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Calibri", color: BLUE },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] },
  ] },
  sections: [
    { properties: { page: { size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      children: titlePage },
    { properties: { page: { size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      footers: { default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "AI-Powered Telecom Fraud, Phishing & SMS Spam Detection  |  Page ", size: 16, color: GREY }),
                   new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GREY })] })] }) },
      children: [...toc, ...body] },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("reports/Project_Report.docx", buf);
  console.log("Wrote reports/Project_Report.docx (" + (buf.length / 1024).toFixed(0) + " KB)");
});
