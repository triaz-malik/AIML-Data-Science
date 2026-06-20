// Generate the business report as a Word (.docx) document.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageBreak, Footer, Header, PageNumber, TableOfContents,
} = require("docx");

const ROOT = __dirname;
const FIG = path.join(ROOT, "outputs", "figures");
const REP = path.join(ROOT, "outputs", "reports");
const OUT = path.join(ROOT, "Telecom_Segmentation_Report.docx");

const NAVY = "1A4D7A";
const ACCENT = "0B7285";
const GREY = "555555";
const LIGHT = "EEF3F8";
const HILITE = "D3F0E0";
const CONTENT_W = 9360; // US Letter, 1" margins

// ---- helpers --------------------------------------------------------------
function readCsv(file) {
  const txt = fs.readFileSync(path.join(REP, file), "utf8").trim();
  const [head, ...rows] = txt.split(/\r?\n/);
  const cols = head.split(",");
  return rows.map((r) => {
    const cells = r.split(",");
    const o = {};
    cols.forEach((c, i) => (o[c || "_name"] = cells[i]));
    return o;
  });
}

function img(file, widthPx) {
  const data = fs.readFileSync(path.join(FIG, file + ".png"));
  // read PNG dimensions from IHDR
  const w = data.readUInt32BE(16);
  const h = data.readUInt32BE(20);
  const height = Math.round((widthPx * h) / w);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 40 },
    children: [
      new ImageRun({
        type: "png",
        data,
        transformation: { width: widthPx, height },
        altText: { title: file, description: file, name: file },
      }),
    ],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: [new TextRun({ text, italics: true, size: 17, color: GREY })],
  });
}

function body(runs, opts = {}) {
  const children = Array.isArray(runs) ? runs : [new TextRun(runs)];
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 120 },
    children,
    ...opts,
  });
}

function bullet(runs) {
  const children = Array.isArray(runs) ? runs : [new TextRun(runs)];
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children });
}

function t(text, o = {}) { return new TextRun({ text, ...o }); }
function b(text, o = {}) { return new TextRun({ text, bold: true, ...o }); }

const cellBorder = { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC" };
const borders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

function cell(text, width, { header = false, fill = null, bold = false, align = AlignmentType.CENTER } = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text, bold: bold || header, color: header ? "FFFFFF" : "000000", size: 18 })],
    })],
  });
}

function table(headerRow, dataRows, widths, hiliteIdx = -1) {
  const rows = [
    new TableRow({
      tableHeader: true,
      children: headerRow.map((h, i) => cell(h, widths[i], { header: true, fill: NAVY, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })),
    }),
  ];
  dataRows.forEach((r, ri) => {
    const isHi = ri === hiliteIdx;
    const fill = isHi ? HILITE : ri % 2 ? LIGHT : "FFFFFF";
    rows.push(new TableRow({
      children: r.map((c, i) => cell(c, widths[i], { fill, bold: isHi, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })),
    }));
  });
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows });
}

// ---- data -----------------------------------------------------------------
const segRows = readCsv("phase3_segment_summary.csv");
const compRows = readCsv("phase5_model_comparison.csv");
const shapRows = readCsv("phase6_shap_importance.csv");

const segOrder = ["High Risk Users", "Premium Users", "Voice Heavy Users", "International Users", "Low Revenue Users"];
const segMap = Object.fromEntries(segRows.map((r) => [r.Segment, r]));
const segTable = segOrder.map((name) => {
  const r = segMap[name];
  return [name, String(Math.round(+r.customers)), `${Math.round(+r.churn_rate * 100)}%`, `$${(+r.avg_charges).toFixed(2)}`, (+r.avg_value).toFixed(1)];
});

let bestModel = "", bestAuc = -1;
compRows.forEach((r) => { if (+r["Test ROC-AUC"] > bestAuc) { bestAuc = +r["Test ROC-AUC"]; bestModel = r._name; } });
const compTable = compRows.map((r) => [r._name, (+r["CV ROC-AUC"]).toFixed(3), (+r["Test ROC-AUC"]).toFixed(3), (+r["Test F1"]).toFixed(3)]);
const compHiliteIdx = compRows.findIndex((r) => r._name === bestModel);

const shapTop = shapRows.slice(0, 6).map((r, i) => [String(i + 1), r._name, (+r.mean_abs_shap).toFixed(3)]);

// ---- document -------------------------------------------------------------
const heading = (text, level) => new Paragraph({ heading: level, children: [new TextRun(text)] });

const doc = new Document({
  creator: "triaz.malik",
  title: "Telecom Customer Segmentation - Business Report",
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: NAVY, font: "Calibri" },
        paragraph: { spacing: { before: 260, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: ACCENT, font: "Calibri" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Telecom Customer Segmentation & Package Recommendation System  —  Page ", size: 16, color: GREY }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GREY })],
      })] }),
    },
    children: [
      // ---- Title page ----
      new Paragraph({ spacing: { before: 2400 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Telecom Customer Segmentation", bold: true, size: 48, color: NAVY })] }),
      new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "& Package Recommendation System", bold: true, size: 48, color: NAVY })] }),
      new Paragraph({ spacing: { before: 200 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Business Report — Methodology, Results & Value", size: 26, color: GREY })] }),
      new Paragraph({ spacing: { before: 600 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Dataset: Orange / BigML Telecom Churn — 3,333 subscribers  •  Overall churn rate 14.5%", size: 21 })] }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- TOC ----
      heading("Contents", HeadingLevel.HEADING_1),
      new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- 1. Executive Summary ----
      heading("1. Executive Summary", HeadingLevel.HEADING_1),
      body([t("This project turns a raw telecom subscriber table into an actionable customer-intelligence system. We profiled 3,333 customers, engineered value and usage features, grouped customers into five business segments, built a recommendation engine that suggests a better package for each subscriber, and trained a churn-prediction model that flags at-risk customers before they leave — with an explainability layer that shows "), t("why", { italics: true }), t(" each prediction is made.")]),
      heading("Headline results", HeadingLevel.HEADING_2),
      bullet([b("Churn prediction: "), t("XGBoost achieves "), b("0.924 ROC-AUC"), t(" and "), b("87% recall"), t(" on churners (holdout test) — most leavers are caught.")]),
      bullet([b("Segmentation: "), t("five clear segments; the "), b("High Risk"), t(" segment (728 customers) churns at "), b("32%"), t(" while paying the "), b("highest bills"), t(" ($72.80 avg) — the priority retention target.")]),
      bullet([b("Recommendations: "), t("a cross-sell / upsell opportunity is flagged for "), b("~40%"), t(" of subscribers.")]),
      bullet([b("Top churn drivers"), t(" (SHAP): total charges, voicemail plan, international calls, and customer-service calls.")]),

      // ---- 2. What Was Done ----
      heading("2. What Was Done", HeadingLevel.HEADING_1),
      body("The work is organised as a reproducible 7-phase pipeline:"),
      table(["Phase", "Activity", "Output"], [
        ["1. EDA", "Distributions, correlations, churn boxplots", "7 charts + insights"],
        ["2. Feature Engineering", "Total usage/charges, intl ratio, value score, segments", "Enriched dataset"],
        ["3. Segmentation", "K-Means + DBSCAN, profiled & named clusters", "5 business segments"],
        ["4. Recommendation", "KNN — 5 most-similar customers per subscriber", "Package per customer"],
        ["5. Churn Prediction", "KNN / RF / XGBoost + SMOTE + tuned CV", "Best model (XGBoost)"],
        ["6. Explainability", "SHAP on the tuned model", "Ranked churn drivers"],
        ["7. Business Value", "Translate analytics into operator actions", "This report"],
      ], [1900, 4500, 2960]),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- 3. EDA ----
      heading("3. Key Findings (EDA)", HeadingLevel.HEADING_1),
      bullet([b("Heavy users"), t(" (top 10% by minutes) churn "), b("47%"), t(" vs 14.5% overall.")]),
      bullet([b("Top payers"), t(" (top 10% by charges) churn "), b("63%"), t(" — the most valuable customers are the most likely to leave.")]),
      bullet([t("Customers with "), b("≥4 service calls"), t(" churn "), b("52%"), t(" — a clear distress signal.")]),
      bullet([b("International-plan"), t(" holders churn "), b("42%"), t(" vs 11.5% without.")]),
      img("01_churn_distribution", 300),
      caption("Churn distribution (14.5% of customers churn)"),
      img("05_customer_service_calls_distribution", 360),
      caption("Churn rises sharply with the number of customer-service calls"),
      img("07_boxplots_churn_vs_retained", 540),
      caption("Churned vs retained customers across key features"),

      // ---- 4. Segments ----
      new Paragraph({ children: [new PageBreak()] }),
      heading("4. Customer Segments", HeadingLevel.HEADING_1),
      body("K-Means (with a DBSCAN cross-check) groups customers into five segments, each mapped to a business profile:"),
      table(["Segment", "Customers", "Churn", "Avg Bill", "Value Score"], segTable, [2600, 1690, 1690, 1690, 1690], 0),
      body([t("The "), b("High Risk Users"), t(" segment (highlighted) is the standout concern: they pay the most yet churn at 32% — protecting them protects the most revenue.")]),
      img("09_segments_pca_scatter", 460),
      caption("Customer segments (K-Means) — PCA projection"),

      // ---- 5. Recommendation ----
      new Paragraph({ children: [new PageBreak()] }),
      heading("5. Package Recommendation Engine", HeadingLevel.HEADING_1),
      body([t("For every subscriber the engine finds the 5 most similar customers — by usage, charges, international calls, tenure and service calls — and recommends the package most common among them: "), t("“Customers similar to you are using Package X.”", { italics: true }), t(" This surfaces a cross-sell / upsell opportunity for roughly "), b("40% of the base"), t(", and gives front-line staff a concrete next-best offer for each customer.")]),

      // ---- 6. Churn results ----
      heading("6. Churn Prediction Results", HeadingLevel.HEADING_1),
      body("Three models were tuned with SMOTE (to handle the 14.5% imbalance) and Stratified 5-fold cross-validation, then scored on the untouched 20% holdout split."),
      table(["Model", "CV ROC-AUC", "Test ROC-AUC", "Test F1"], compTable, [2760, 2200, 2200, 2200], compHiliteIdx),
      body([b("XGBoost"), t(" is the best model: "), b("0.924 ROC-AUC"), t(", 95% accuracy, and "), b("87% recall on churners"), t(" — it catches the large majority of customers who are about to leave.")]),
      img("12_roc_curves", 330),
      caption("ROC curves for the three tuned models"),
      img("13_confusion_matrix", 280),
      caption("Confusion matrix — best model (XGBoost) on holdout"),

      // ---- 7. SHAP ----
      new Paragraph({ children: [new PageBreak()] }),
      heading("7. Why Customers Churn (SHAP)", HeadingLevel.HEADING_1),
      body("SHAP explains the model's predictions. The strongest churn drivers are:"),
      table(["Rank", "Driver", "Impact (mean |SHAP|)"], shapTop, [1400, 4960, 3000]),
      img("15_shap_importance_bar", 470),
      caption("Global feature importance (mean absolute SHAP value)"),

      // ---- 8. Business value ----
      new Paragraph({ children: [new PageBreak()] }),
      heading("8. Business Value", HeadingLevel.HEADING_1),
      body("The pipeline converts raw subscriber data into five concrete operator capabilities:"),
      bullet([b("Identify high-value customers"), t(" — the Customer Value Score and segments rank the base by worth.")]),
      bullet([b("Recommend better packages"), t(" — the KNN engine gives every subscriber a next-best-offer, with upsell flags.")]),
      bullet([b("Reduce churn"), t(" — every subscriber gets a churn probability and risk band, so retention teams act "), t("before", { italics: true }), t(" customers leave.")]),
      bullet([b("Increase ARPU"), t(" — move Low-Revenue users onto fitting paid packages and upsell Premium / International users.")]),
      bullet([b("Sharper retention campaigns"), t(" — target the "), b("High-Risk & High-Value"), t(" overlap, where saved revenue per customer is greatest.")]),
      heading("Recommended actions", HeadingLevel.HEADING_2),
      bullet("Launch a proactive save-desk for High-Risk customers with ≥3 service calls."),
      bullet("Audit the international plan — its holders churn ~4× the base; revisit pricing / quality."),
      bullet([t("Feed the scored dataset ("), t("telecom_scored.csv", { font: "Consolas" }), t(") into a Power BI dashboard for ongoing monitoring.")]),
      new Paragraph({ spacing: { before: 200 }, children: [new TextRun({ text: "All figures in this report are produced by the reproducible pipeline; the scored dataset is Power BI-ready.", italics: true, size: 18, color: GREY })] }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`Wrote ${OUT} (${Math.round(buf.length / 1024)} KB)`);
});
