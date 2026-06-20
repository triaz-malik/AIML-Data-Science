const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, LevelFormat, TableOfContents,
  HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign,
  PageNumber, PageBreak,
} = require("docx");

const PLOTS = "outputs/plots";
const CW = 9360; // content width (US Letter, 1" margins)

// ---- helpers ----------------------------------------------------------
const ACCENT = "2E5E4E";
const ACCENT2 = "1F5673";

function img(file, w) {
  const sizes = {
    "01_target_distribution.png": [1412, 477], "02_hour_vs_energy.png": [1412, 477],
    "03_temperature_vs_energy.png": [1412, 477], "04_humidity_vs_energy.png": [1412, 477],
    "05a_correlation_heatmap.png": [1403, 1192], "05b_target_correlations.png": [862, 972],
    "06a_daily_trend.png": [1410, 422], "06b_weekday_hour_heatmap.png": [1293, 477],
    "07_model_comparison.png": [1408, 477], "08_actual_vs_predicted.png": [1411, 422],
    "09a_shap_importance.png": [862, 807], "09b_shap_beeswarm.png": [826, 807],
  };
  const [ow, oh] = sizes[file];
  const width = w, height = Math.round(w * oh / ow);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(path.join(PLOTS, file)),
      transformation: { width, height },
      altText: { title: file, description: file, name: file },
    })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text, italics: true, size: 18, color: "666666" })],
  });
}

function h1(text) { return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] }); }
function h2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] }); }
function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text, ...opts })] });
}
function runs(children) { return new Paragraph({ spacing: { after: 120 }, children }); }
function bullet(text, level = 0) {
  return new Paragraph({ numbering: { reference: "b", level }, spacing: { after: 60 },
    children: typeof text === "string" ? [new TextRun(text)] : text });
}
function numbered(text) {
  return new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 60 },
    children: typeof text === "string" ? [new TextRun(text)] : text });
}

const thinB = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: thinB, bottom: thinB, left: thinB, right: thinB };
function cell(content, { w, head = false, fill, bold = false, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 120, right: 120 },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align, children: [
      new TextRun({ text: content, bold: head || bold, color: head ? "FFFFFF" : "000000", size: 20 }),
    ] })],
  });
}
function table(widths, headRow, dataRows, headFill = ACCENT) {
  const rows = [new TableRow({ tableHeader: true, children: headRow.map((t, i) =>
    cell(t, { w: widths[i], head: true, fill: headFill, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) })];
  dataRows.forEach((r, ri) => {
    rows.push(new TableRow({ children: r.map((t, i) =>
      cell(String(t), { w: widths[i], fill: ri % 2 ? "F2F6F4" : undefined,
        bold: i === 0, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) }));
  });
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, rows });
}
function spacer(after = 120) { return new Paragraph({ spacing: { after }, children: [new TextRun("")] }); }

// ---- document ---------------------------------------------------------
const children = [];

// Title page
children.push(
  new Paragraph({ spacing: { before: 2600, after: 0 }, alignment: AlignmentType.CENTER,
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 8 } },
    children: [new TextRun({ text: "Energy Consumption Prediction", bold: true, size: 56, color: ACCENT })] }),
  new Paragraph({ spacing: { before: 240 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Predicting & Explaining Household Appliance Energy Use (Wh)", size: 30, color: "444444" })] }),
  new Paragraph({ spacing: { before: 160 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "A Machine Learning Case Study  •  EDA → Modelling → SHAP → Business Value", size: 22, color: "777777" })] }),
  new Paragraph({ spacing: { before: 1400 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Dataset: UCI / Kaggle Appliances Energy Prediction", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "19,735 records · 10-minute cadence · Jan–May 2016", size: 22, color: "777777" })] }),
  new Paragraph({ spacing: { before: 1600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Prepared June 2026", size: 20, color: "999999" })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// TOC
children.push(h1("Contents"),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }));

// 1. Executive summary
children.push(h1("1. Executive Summary"));
children.push(p("Electricity costs are rising and energy use is uneven across the day. This project builds a machine-learning system that predicts a household's next-interval appliance energy consumption and, just as importantly, explains what drives it — so homeowners, building operators and utilities can shift load to cheaper hours, optimise HVAC, and forecast demand peaks.", { size: 22 }));
children.push(runs([
  new TextRun({ text: "Headline result: ", bold: true }),
  new TextRun("a tuned "),
  new TextRun({ text: "LightGBM", bold: true }),
  new TextRun(" model explains "),
  new TextRun({ text: "57% of the variance", bold: true }),
  new TextRun(" in energy use on a held-out future period, with a typical error of about "),
  new TextRun({ text: "24 Wh", bold: true }),
  new TextRun(" — and SHAP reveals that time-of-day rhythm and recent usage, more than weather, drive this home's consumption."),
]));
children.push(h2("Key results at a glance"));
children.push(table([3120, 2080, 2080, 2080],
  ["Model", "Test R²", "RMSE (Wh)", "MAE (Wh)"],
  [["Linear Regression", "0.40", "70.1", "28.1"],
   ["Random Forest", "0.56", "60.2", "27.9"],
   ["XGBoost", "0.56", "60.1", "23.6"],
   ["LightGBM  (best)", "0.57", "59.3", "23.7"]]));
children.push(spacer());
children.push(runs([
  new TextRun({ text: "An honest number. ", bold: true, color: ACCENT2 }),
  new TextRun("Many notebooks claim 80–90% R² on this dataset, but that comes from a random train/test split that leaks the autocorrelated past into the future. This project evaluates on a strict "),
  new TextRun({ text: "chronological split", italics: true }),
  new TextRun(" with lag features that only ever see history — so 0.57 is what you would actually get forecasting truly unseen data. It matches the published benchmark (Candanedo et al., 2017)."),
]));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 2. Business problem
children.push(h1("2. Business Problem"));
children.push(p("Homeowners, smart-building operators and utility companies all want to predict future energy consumption, understand what causes high consumption, and act on it to reduce cost and strain on the grid."));
children.push(h2("Business questions"));
[ "What drives appliance energy consumption?",
  "Can we predict the next interval's energy usage?",
  "Which environmental and behavioural factors impact energy use most?",
  "Where can loads be shifted or HVAC tuned to save cost?",
].forEach(t => children.push(bullet(t)));
children.push(h2("Who benefits"));
children.push(table([3120, 6240],
  ["Stakeholder", "Value delivered"],
  [["Homeowners", "Lower bills via load shifting and smarter HVAC scheduling"],
   ["Utilities", "Demand-peak forecasting for better grid planning, fewer overloads"],
   ["Building managers", "Optimised HVAC schedules, lower operating cost & carbon footprint"]], ACCENT2));

// 3. Dataset
children.push(h1("3. Dataset Overview"));
children.push(p("The dataset records a single house over roughly 4.5 months at a perfectly regular 10-minute cadence. It has zero missing values; rv1/rv2 are identical random-noise columns excluded from modelling."));
children.push(table([2400, 6960],
  ["Type", "Examples"],
  [["Target", "Appliances energy use (Wh)"],
   ["Indoor sensors", "T1–T9 temperatures across 9 zones"],
   ["Humidity", "RH_1–RH_9 per zone"],
   ["Outdoor / weather", "T_out, Pressure, Wind speed, Visibility, Tdewpoint, RH_out"],
   ["Time", "Date-time → hour, day, month, weekend, night flags"]], ACCENT2));
children.push(spacer());
children.push(runs([
  new TextRun({ text: "Target is heavily right-skewed ", bold: true }),
  new TextRun("(skew ≈ 3.4; median 60 Wh, max 1080 Wh). Most intervals are quiet with occasional consumption bursts — which is why we model log1p(Appliances) and keep the spikes rather than deleting them."),
]));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 4. EDA
children.push(h1("4. Exploratory Data Analysis"));
children.push(p("Six views answer: how is energy distributed, when is it used, and what is it related to?"));

children.push(h2("4.1  Energy consumption distribution"));
children.push(img("01_target_distribution.png", 600));
children.push(caption("Raw target (left) is sharply right-skewed; log1p transform (right) is near-symmetric — the modelling target."));
children.push(p("Insight: a few high-consumption periods dominate the tail. Modelling the log keeps those peaks while stabilising the error."));

children.push(h2("4.2  Hour of day vs energy"));
children.push(img("02_hour_vs_energy.png", 600));
children.push(caption("Mean energy by hour (left) and distribution per hour (right)."));
children.push(p("Insight: usage is lowest overnight, rises through the morning, and peaks in the early evening (~17–20h). This is the prime window for load-shifting recommendations."));

children.push(h2("4.3  Temperature vs energy (HVAC dependency)"));
children.push(img("03_temperature_vs_energy.png", 600));
children.push(caption("Binned mean energy vs outdoor temperature (left); living-room temperature scatter (right)."));
children.push(p("Insight: the temperature relationship is real but non-linear — evidence that tree models will outperform linear regression."));

children.push(h2("4.4  Humidity vs energy"));
children.push(img("04_humidity_vs_energy.png", 600));
children.push(caption("Mean energy across indoor (left) and outdoor (right) humidity bins."));
children.push(p("Insight: humidity shows a mild, non-monotonic association with energy — a secondary HVAC signal."));

children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h2("4.5  Correlation structure & strongest linear drivers"));
children.push(img("05a_correlation_heatmap.png", 470));
children.push(caption("Correlation heatmap — room temperatures move together (shared building climate)."));
children.push(img("05b_target_correlations.png", 360));
children.push(caption("Each feature's linear correlation with appliance energy."));
children.push(p("Insight: individual linear correlations with energy are modest, confirming that value comes from non-linear interactions and temporal context — not any single sensor."));

children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h2("4.6  Temporal trend & weekly seasonality"));
children.push(img("06a_daily_trend.png", 600));
children.push(caption("Daily mean energy across the recording period."));
children.push(img("06b_weekday_hour_heatmap.png", 600));
children.push(caption("Mean energy by weekday × hour — the recurring evening hot-spot."));
children.push(p("Insight: a strong, repeating daily pattern (hot evenings, cool nights) justifies the time-based and lag features used in modelling."));

// 5. Cleaning + FE
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h1("5. Data Cleaning & Feature Engineering"));
children.push(h2("Cleaning"));
children.push(bullet("Missing values: none — confirmed with an explicit audit, so no imputation needed."));
children.push(bullet("Outliers: ~11% of intervals exceed the IQR fence, but these are genuine demand peaks. We keep them (modelling log1p already compresses the tail) instead of winsorizing away the events the business cares about."));
children.push(bullet("Noise columns rv1/rv2 excluded; SHAP later confirms the model ignores them."));
children.push(h2("Engineered features"));
children.push(table([3000, 6360],
  ["Feature", "Why it matters"],
  [["hour, is_weekend, is_night, month", "Capture daily & weekly behaviour"],
   ["hour_sin, hour_cos", "Cyclical encoding so 23:00 and 00:00 are neighbours"],
   ["indoor_temp_mean", "Overall house warmth (rooms are correlated)"],
   ["temp_diff (indoor − outdoor)", "The key HVAC heating/cooling driver"],
   ["lag_1 / lag_2 / lag_3", "Energy is strongly autocorrelated — recent past predicts next step"],
   ["roll_mean_3/6/18, roll_std_6", "Short-to-medium consumption trend & volatility (30 min → 3 h)"]], ACCENT2));
children.push(spacer());
children.push(runs([
  new TextRun({ text: "Leakage-safe by construction: ", bold: true, color: ACCENT2 }),
  new TextRun("every lag/rolling feature uses .shift(1), so a row only sees the past. Combined with the chronological split and TimeSeriesSplit tuning, the evaluation reflects real forecasting performance."),
]));

// 6. Models
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h1("6. Modelling & Comparison"));
children.push(p("Four models of increasing power, all predicting log1p(Appliances) and scored on the real Wh scale over the held-out future period. Random Forest and XGBoost were tuned with RandomizedSearchCV; LightGBM with Optuna (25 trials) — all over a TimeSeriesSplit."));
children.push(table([2600, 1500, 1500, 1500, 2260],
  ["Model", "Test R²", "RMSE", "MAE", "Role"],
  [["Linear Regression", "0.40", "70.1", "28.1", "Interpretable baseline"],
   ["Random Forest", "0.56", "60.2", "27.9", "Non-linear interactions"],
   ["XGBoost", "0.56", "60.1", "23.6", "Strong gradient boosting"],
   ["LightGBM", "0.57", "59.3", "23.7", "Best — fast & accurate"]]));
children.push(img("07_model_comparison.png", 600));
children.push(caption("Test R² and RMSE by model — boosting beats the linear baseline by ~17 R² points."));
children.push(img("08_actual_vs_predicted.png", 600));
children.push(caption("LightGBM predictions vs actual over the unseen test period — peaks and rhythm tracked well."));

// 7. Explainability
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h1("7. What Drives Energy — SHAP Explainability"));
children.push(p("SHAP decomposes each prediction into per-feature contributions, answering which variables matter and why a given prediction is high or low."));
children.push(img("09a_shap_importance.png", 380));
children.push(caption("Mean |SHAP| — overall feature importance."));
children.push(img("09b_shap_beeswarm.png", 380));
children.push(caption("SHAP value distribution — direction & magnitude of each feature's effect."));
children.push(runs([
  new TextRun({ text: "Finding: ", bold: true }),
  new TextRun("time-of-day (hour_cos/hour_sin, is_night) and recent usage (lags/rolling means) dominate — this household's consumption is driven more by occupancy rhythm than by weather. Among environmental features, indoor temperatures and lighting carry the most signal. The random columns rv1/rv2 contribute essentially nothing, a healthy sign the model isn't overfitting noise."),
]));

// 8. Business value
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h1("8. Business Value & Recommendations"));
children.push(h2("Actionable recommendations"));
children.push(numbered("Load shifting: schedule deferrable appliances (dishwasher, washing machine, EV charging) into the 02:00–04:00 off-peak window — an estimated 5–15% cost saving with no change in usage."));
children.push(numbered("HVAC optimisation: temp_diff (indoor − outdoor) is a top environmental driver; pre-conditioning before the evening peak flattens the 17–20h load."));
children.push(numbered("Demand-peak forecasting: next-interval predictions let utilities and building managers anticipate evening peaks for better grid planning."));
children.push(numbered("Anomaly watch: flag intervals where actual ≫ predicted as possible faulty appliances or unexpected spikes."));
children.push(h2("Quantified value"));
children.push(table([3120, 6240],
  ["Lever", "Expected impact"],
  [["Smart-home load shifting", "5–15% electricity-cost reduction"],
   ["HVAC schedule optimisation", "Lower evening peak demand & operating cost"],
   ["Utility demand forecasting", "Fewer overloads, better capacity planning"],
   ["Residual anomaly alerts", "Early detection of faulty appliances"]], ACCENT2));

// 9. Deliverables + future
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h1("9. Deliverables & Next Steps"));
children.push(h2("What was produced"));
children.push(bullet("Executed Jupyter notebook (Energy_Consumption_Prediction.ipynb) + HTML report — full narrative, all plots, models and SHAP."));
children.push(bullet("Reusable pipeline script (energy_pipeline.py)."));
children.push(bullet("12 figures, saved model (best_model.joblib), metrics table, and a dashboard-ready predictions CSV."));
children.push(bullet("Auto-generated recommendations (recommendations.md) and this report."));
children.push(h2("Future improvements"));
children.push(bullet("Time-series models: benchmark Prophet / LSTM / GRU for multi-step forecasts."));
children.push(bullet("Weather-forecast integration: feed tomorrow's forecast to predict next-day demand."));
children.push(bullet("Anomaly detection: Isolation Forest / autoencoder on residuals for faulty-appliance alerts."));
children.push(bullet("Reinforcement learning: an agent that schedules appliances to minimise cost under a tariff."));
children.push(bullet("Power BI / Tableau dashboard built on predictions_for_dashboard.csv — forecast vs actual, peak hours, savings opportunities."));

// ---- assemble ---------------------------------------------------------
const doc = new Document({
  creator: "Energy Consumption Prediction Project",
  title: "Energy Consumption Prediction — ML Case Study",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCD8D2", space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Arial", color: ACCENT2 },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "b", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 600, hanging: 280 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1100, hanging: 280 } } } } ] },
      { reference: "n", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 600, hanging: 320 } } } } ] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "DDDDDD", space: 4 } },
      children: [new TextRun({ text: "Energy Consumption Prediction", size: 16, color: "999999" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Page ", size: 16, color: "999999" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "999999" }),
        new TextRun({ text: " of ", size: 16, color: "999999" }),
        new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: "999999" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("Energy_Consumption_Prediction_Report.docx", buf);
  console.log("WROTE Energy_Consumption_Prediction_Report.docx", buf.length, "bytes");
});
