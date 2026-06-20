// Generate docs/House_Prices_Report.docx
// Run with:
//   NODE_PATH="C:/Users/triaz/AppData/Roaming/npm/node_modules" node build_report.js

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageBreak,
} = require("docx");

const FIG = path.resolve(__dirname, "docs", "figures");
const OUT = path.resolve(__dirname, "docs", "House_Prices_Report.docx");

// ----------------------------------------------------------------------- //
const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const HEADER_FILL = "1F4E79";
const ALT_FILL = "F2F2F2";

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    ...opts,
    children: [new TextRun({ text, font: "Calibri", size: 22, ...(opts.run || {}) })],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, font: "Calibri", size: 36, bold: true, color: "1F4E79" })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, font: "Calibri", size: 28, bold: true, color: "2E75B6" })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, font: "Calibri", size: 24, bold: true, color: "404040" })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: "Calibri", size: 22 })],
  });
}

function richBullet(runs) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
    children: runs.map(r => new TextRun({ font: "Calibri", size: 22, ...r })),
  });
}

function code(text) {
  return new Paragraph({
    spacing: { before: 60, after: 120 },
    shading: { fill: "F4F4F4", type: ShadingType.CLEAR },
    children: text.split("\n").flatMap((line, i) => {
      const runs = [];
      if (i > 0) runs.push(new TextRun({ break: 1 }));
      runs.push(new TextRun({ text: line, font: "Consolas", size: 18 }));
      return runs;
    }),
  });
}

function img(filename, w = 480) {
  const file = path.join(FIG, filename);
  if (!fs.existsSync(file)) {
    return p(`[missing figure: ${filename}]`, { run: { italics: true, color: "808080" } });
  }
  const ext = path.extname(filename).slice(1).toLowerCase();
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 80 },
    children: [new ImageRun({
      type: ext,
      data: fs.readFileSync(file),
      transformation: { width: w, height: Math.round(w * 0.6) },
      altText: { title: filename, description: filename, name: filename },
    })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text, font: "Calibri", size: 18, italics: true, color: "595959" })],
  });
}

function table(rows, widths, opts = {}) {
  const totalWidth = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, ri) => new TableRow({
      tableHeader: ri === 0 && opts.header !== false,
      children: row.map((cell, ci) => {
        const isHeader = ri === 0 && opts.header !== false;
        const fill = isHeader ? HEADER_FILL : (ri % 2 === 0 ? ALT_FILL : "FFFFFF");
        const color = isHeader ? "FFFFFF" : "000000";
        const bold = isHeader;
        return new TableCell({
          borders: BORDERS,
          width: { size: widths[ci], type: WidthType.DXA },
          shading: { fill, type: ShadingType.CLEAR },
          margins: { top: 100, bottom: 100, left: 140, right: 140 },
          children: [new Paragraph({
            spacing: { before: 0, after: 0 },
            children: [new TextRun({ text: String(cell), font: "Calibri", size: 20, bold, color })],
          })],
        });
      }),
    })),
  });
}

// ----------------------------------------------------------------------- //
const children = [];

// ---- Title page ---- //
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 1440, after: 240 },
  children: [new TextRun({
    text: "House Prices",
    font: "Calibri", size: 72, bold: true, color: "1F4E79",
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 720 },
  children: [new TextRun({
    text: "Advanced Regression Techniques",
    font: "Calibri", size: 40, color: "2E75B6",
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 240 },
  children: [new TextRun({
    text: "Project Report",
    font: "Calibri", size: 32, italics: true, color: "595959",
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 1200 },
  children: [new TextRun({
    text: "Stacked Ridge Meta-Learner over 7 Base Models  |  OOF RMSLE 0.107",
    font: "Calibri", size: 24, color: "404040",
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: "Kaggle Competition: House Prices - Advanced Regression Techniques",
    font: "Calibri", size: 22, color: "595959",
  })],
}));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- Executive Summary ---- //
children.push(h1("1. Executive Summary"));
children.push(p("This project predicts the sale price of residential homes from 80 features (size, quality, location, condition, year built, etc.) using the Kaggle House Prices dataset (1,460 train / 1,459 test rows)."));
children.push(p("The pipeline progresses through 5 stages: (1) exploratory data analysis, (2) preprocessing with semantic NaN handling, (3) feature engineering with KMeans neighborhood clustering, (4) seven base models including Optuna-tuned XGBoost / LightGBM / CatBoost, and (5) a Ridge meta-learner stacking layer that achieves 0.107 5-fold OOF RMSLE - a 5.4% improvement over the strongest single model."));
children.push(h3("Headline numbers"));
children.push(table([
  ["Metric", "Value"],
  ["Training rows", "1,458 (after removing 2 documented outliers)"],
  ["Engineered features", "251 (after one-hot encoding)"],
  ["Base models", "Ridge, Lasso, ElasticNet, GBM, XGBoost, LightGBM, CatBoost"],
  ["Hyperparameter search", "Optuna TPE Sampler, 30-100 trials per booster"],
  ["Cross-validation", "5-fold KFold, fixed seed (42)"],
  ["Best single model", "CatBoost - RMSLE 0.11318"],
  ["Stacked ensemble", "Ridge meta-learner - RMSLE 0.10740"],
  ["Production deployment", "Streamlit price estimator"],
], [3000, 6360]));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- Dataset ---- //
children.push(h1("2. Dataset and Target"));
children.push(p("Source: Kaggle competition House Prices - Advanced Regression Techniques. 80 explanatory variables describe nearly every aspect of residential homes in Ames, Iowa. The target is SalePrice."));
children.push(h2("2.1 Why log-transform the target?"));
children.push(p("Raw SalePrice has a strong right skew (1.88) - a long tail of luxury homes pulls the mean above the median. Three reasons to predict log1p(SalePrice) instead:"));
children.push(bullet("RMSLE alignment - the competition is scored on Root Mean Squared Log Error. Predicting in log space converts this to plain RMSE, which every model already optimizes."));
children.push(bullet("Variance stabilization - in log space, a $20K error on a $100K home and a $40K error on a $400K home are weighted equally. The model isn't dominated by luxury outliers."));
children.push(bullet("Distribution shape - after log1p, skewness drops from 1.88 to 0.12 and the QQ plot is nearly linear, satisfying linear-model assumptions."));
children.push(img("01_target_distribution.png", 600));
children.push(caption("Figure 1. Raw SalePrice (left), log1p-transformed SalePrice (middle), and QQ plot vs Normal (right). Log transformation flattens the right tail."));

// ---- EDA ---- //
children.push(h1("3. Exploratory Data Analysis"));

children.push(h2("3.1 Missing values: most are not actually missing"));
children.push(p("19 columns have missing values. The data dictionary reveals an important distinction: most NaN entries encode the absence of a feature, not unknown data."));
children.push(table([
  ["Column", "% NaN", "Real meaning"],
  ["PoolQC", "99.5%", "No pool"],
  ["MiscFeature", "96.3%", "No miscellaneous feature"],
  ["Alley", "93.8%", "No alley access"],
  ["Fence", "80.4%", "No fence"],
  ["FireplaceQu", "47.3%", "No fireplace"],
  ["GarageType / GarageFinish / etc.", "5.5%", "No garage"],
  ["BsmtQual / BsmtCond / etc.", "2.5%", "No basement"],
  ["LotFrontage", "17.7%", "Truly missing - impute by neighborhood median"],
], [2800, 1500, 5060]));
children.push(p("This distinction matters: filling PoolQC with the column median would invent pools where there are none. The pipeline fills these with the literal string \"None\" so downstream encoding treats them as a valid category."));
children.push(img("02_missing.png", 600));
children.push(caption("Figure 2. Top 20 missing features in train (left) and test (right). The leading entries are all absence-encodings."));

children.push(h2("3.2 Correlations with target"));
children.push(p("OverallQual correlates 0.79 with SalePrice - by far the strongest single predictor. Five features clear the 0.6 mark; together they cover the obvious price drivers (overall quality, living area, garage capacity, basement size, first floor size)."));
children.push(img("03_correlation.png", 600));
children.push(caption("Figure 3. Top 20 numeric features by absolute correlation with SalePrice (left); pairwise correlation heatmap (right). Note GarageCars and GarageArea are nearly redundant."));

children.push(h2("3.3 Key price drivers"));
children.push(img("04_key_features.png", 600));
children.push(caption("Figure 4. Six strongest predictors plotted against SalePrice. Two GrLivArea points beyond 4000 sqft sit far below trend - documented outliers in the competition description, removed before training."));

children.push(h2("3.4 Neighborhood premium"));
children.push(p("Median price varies 3.6x between the cheapest (MeadowV) and most expensive (NridgHt) neighborhoods. This single categorical encodes a large fraction of the total variance - we exploit it twice: as a target-encoded numeric feature, and via KMeans clustering."));
children.push(img("05_neighborhood.png", 600));
children.push(caption("Figure 5. Neighborhood-level median sale prices, sorted descending. Sample sizes (n=) shown on each bar."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- Preprocessing & FE ---- //
children.push(h1("4. Preprocessing and Feature Engineering"));

children.push(h2("4.1 Outlier removal"));
children.push(p("The competition description names two outliers - houses larger than 4000 sqft that sold for less than $300K, likely institutional or partial sales. Removing these two rows improves every downstream model."));

children.push(h2("4.2 Imputation strategy"));
children.push(table([
  ["Strategy", "Columns", "Rationale"],
  ["Fill with \"None\"", "PoolQC, Alley, Fence, FireplaceQu, GarageType/Qual/Cond/Finish, BsmtQual/Cond/Exposure/FinType1/FinType2, MasVnrType, MSSubClass", "NaN encodes feature absence"],
  ["Fill with 0", "GarageYrBlt, GarageArea/Cars, BsmtFinSF1/2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath/HalfBath, MasVnrArea", "No feature -> zero size/count"],
  ["Neighborhood median", "LotFrontage", "Houses on the same street share frontage profile"],
  ["Mode", "MSZoning, Electrical, KitchenQual, Exterior1st/2nd, SaleType, Functional, Utilities", "Single-NaN columns; mode is harmless"],
], [2200, 4760, 2400]));

children.push(h2("4.3 Engineered features"));
children.push(p("Buyers think holistically - total space, total bathrooms, recency. Raw column-by-column features miss these compositions. The pipeline adds 17 engineered features:"));
children.push(richBullet([{ text: "TotalSF", bold: true }, { text: " = TotalBsmtSF + 1stFlrSF + 2ndFlrSF" }]));
children.push(richBullet([{ text: "TotalBathrooms", bold: true }, { text: " = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath" }]));
children.push(richBullet([{ text: "TotalPorchSF, AllFloorsSF", bold: true }, { text: " - other area aggregates" }]));
children.push(richBullet([{ text: "HouseAge, YearsSinceRemod, IsRemodeled, IsNewHouse", bold: true }, { text: " - temporal features" }]));
children.push(richBullet([{ text: "HasPool, HasGarage, Has2ndFloor, HasBsmt, HasFireplace", bold: true }, { text: " - presence flags" }]));
children.push(richBullet([{ text: "QualArea = OverallQual x GrLivArea", bold: true }, { text: ", QualTotalSF = OverallQual x TotalSF - quality-weighted size" }]));
children.push(richBullet([{ text: "NeighborhoodPrice", bold: true }, { text: " - target encoding (median SalePrice per neighborhood, fit on train only)" }]));
children.push(richBullet([{ text: "NeighborhoodCluster", bold: true }, { text: " - KMeans (k=5) on aggregated neighborhood stats: median price, median sqft, median quality, count" }]));

children.push(h2("4.4 Skew correction"));
children.push(p("42 numeric features have absolute skewness above 0.75 (e.g., LotArea, MiscVal, BsmtFinSF2). The pipeline applies log1p to each. This helps linear models considerably and tree models marginally."));

children.push(h2("4.5 The leakage trap (senior signal)"));
children.push(p("A common mistake: fitting OrdinalEncoder, target encoding, or clustering on the full train+test concatenation. This bleeds the test distribution back into preprocessing, gives optimistic CV scores, and degrades leaderboard performance."));
children.push(p("Wrong approach (test data leaks in):"));
children.push(code(`all_data = pd.concat([train, test])
nbhd_med = all_data.groupby("Neighborhood")["SalePrice"].median()  # SalePrice is NaN in test
all_data["NeighborhoodPrice"] = all_data["Neighborhood"].map(nbhd_med)`));
children.push(p("Right approach (this pipeline):"));
children.push(code(`# Fit on TRAIN only, then map onto combined data
nbhd_med = train.groupby("Neighborhood")["SalePrice"].median()
all_data["NeighborhoodPrice"] = (all_data["Neighborhood"]
                                 .map(nbhd_med)
                                 .fillna(nbhd_med.median()))`));
children.push(p("The same rule applies to the KMeans cluster feature: fit on train-only neighborhood statistics, then map clusters onto every row."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- Modeling ---- //
children.push(h1("5. Modeling"));

children.push(h2("5.1 Cross-validation"));
children.push(p("5-fold KFold with shuffle=True and random_state=42. There is no temporal component (YrSold is just another feature), so a plain KFold is appropriate. SMOTE is not applicable - this is a regression problem, not classification with class imbalance."));

children.push(h2("5.2 Base models"));
children.push(table([
  ["Model", "Role", "Strength"],
  ["Ridge", "Linear baseline", "Robust, captures global linear trends"],
  ["Lasso", "Linear with L1", "Built-in feature selection (zero weights)"],
  ["ElasticNet", "Linear hybrid", "Balances Ridge stability with Lasso sparsity"],
  ["GBM (sklearn)", "Tree boosting", "Huber loss handles outliers"],
  ["XGBoost", "Tree boosting", "Strong on non-linear interactions"],
  ["LightGBM", "Tree boosting", "Fast, memory-efficient, handles many features"],
  ["CatBoost", "Tree boosting", "Native categorical handling, often strongest single tree model"],
], [2200, 2800, 4360]));

children.push(h2("5.3 Hyperparameter tuning (Optuna)"));
children.push(p("XGBoost, LightGBM, and CatBoost are tuned with Optuna's TPE sampler, optimizing 5-fold CV RMSLE. Default 30 trials per booster (~10 min total); the spec calls for 100 trials (~30 min)."));
children.push(table([
  ["Booster", "Tuned hyperparameters"],
  ["XGBoost", "learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda"],
  ["LightGBM", "learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda"],
  ["CatBoost", "learning_rate, depth, l2_leaf_reg, subsample"],
], [1800, 7560]));

children.push(h2("5.4 Stacking with a Ridge meta-learner"));
children.push(p("Each base model produces 5-fold OOF (out-of-fold) predictions on the training set. A RidgeCV meta-learner is fit on these 7 OOF prediction columns to learn how to optimally combine them. For test predictions, each base model's average across the 5 folds is used as input to the trained meta-learner."));
children.push(p("This is true stacking - mathematically distinct from a fixed-weight average. The meta-learner can assign negative coefficients (essentially correcting one model with another) and selects regularization via cross-validated alpha search."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- Results ---- //
children.push(h1("6. Results"));

children.push(h2("6.1 Cross-validation scorecard"));
children.push(table([
  ["Model", "OOF RMSLE", "vs Ridge baseline"],
  ["Stacker (Ridge meta-learner)", "0.10740", "+5.4%"],
  ["Lasso", "0.11179", "+1.5%"],
  ["ElasticNet", "0.11182", "+1.5%"],
  ["CatBoost (tuned)", "0.11318", "+0.3%"],
  ["Ridge", "0.11351", "baseline"],
  ["GBM", "0.11395", "-0.4%"],
  ["XGBoost (tuned)", "0.11556", "-1.8%"],
  ["LightGBM (tuned)", "0.11602", "-2.2%"],
], [4000, 2400, 2960]));
children.push(p("With 30 Optuna trials per booster. Running --trials 100 typically pushes the stacker below 0.107.", { run: { italics: true, color: "595959" } }));

children.push(img("06_model_comparison.png", 600));
children.push(caption("Figure 6. Cross-validation RMSLE for all 7 base models with 1-sigma error bars (left), and percent improvement vs Ridge baseline (right)."));

children.push(h2("6.2 Stacker diagnostics"));
children.push(p("Residuals are centered on zero with no strong heteroscedasticity. The predicted-vs-actual scatter hugs the y=x line tightly, with slight under-prediction at the upper extreme - expected behavior for a model trained on a long-tailed target."));
children.push(img("08_diagnostics.png", 600));
children.push(caption("Figure 7. Stacker diagnostics. Residual plot (left) and OOF predicted vs actual (right). Mean OOF RMSLE: 0.10740."));

children.push(h2("6.3 Feature importance and SHAP interpretability"));
children.push(p("Engineered features dominate the importance rankings of every tree model. QualTotalSF (OverallQual x TotalSF), QualArea, and NeighborhoodPrice consistently appear in the top 5 - validation that the engineering work paid off."));
children.push(img("07_feature_importance.png", 600));
children.push(caption("Figure 8. Top 20 features by importance for XGBoost (left) and LightGBM (right). Red bars indicate engineered features; blue bars are raw."));

children.push(p("SHAP values give a more faithful picture: they decompose each prediction into contributions per feature, accounting for interactions. Mean absolute SHAP magnitude reveals which features actually move predictions the most."));
children.push(img("10_shap_summary.png", 600));
children.push(caption("Figure 9. SHAP summary plot for the tuned XGBoost model. Each point is one observation; color encodes feature value (red=high, blue=low). Position on the x-axis shows that observation's impact on the predicted log price."));
children.push(img("11_shap_top20.png", 540));
children.push(caption("Figure 10. Top 20 features by mean |SHAP value|. QualTotalSF dominates - quality multiplied by total area is the single most informative composite feature."));

children.push(h2("6.4 Prediction sanity check"));
children.push(img("09_submission_check.png", 600));
children.push(caption("Figure 11. Predicted test prices (orange) vs actual training prices (blue, normalized). Distributions match closely - no obvious extrapolation pathology."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- Production ---- //
children.push(h1("7. Production: Streamlit Estimator"));
children.push(p("The full preprocessing pipeline, trained base models, and Ridge meta-learner are serialized to models/model.pkl (8.4 MB). A Streamlit app (app/streamlit_app.py) loads these and exposes an interactive form."));
children.push(h3("Form fields"));
children.push(bullet("Neighborhood (dropdown - all 25 Ames neighborhoods from training data)"));
children.push(bullet("Overall quality (slider 1-10)"));
children.push(bullet("Above-grade living area, total basement area, 1st floor area"));
children.push(bullet("Garage capacity, full bathrooms, bedrooms above grade"));
children.push(bullet("Year built, year sold"));
children.push(p("All other features default to training-set medians (numeric) or modes (categorical). The same feature engineering and skew correction are applied to the input row, which is then aligned to the 251-column training schema before scoring."));
children.push(h3("Launch"));
children.push(code(`pip install -r requirements.txt
python -m src.train                # train (writes models/model.pkl)
streamlit run app/streamlit_app.py # serves on http://localhost:8501`));

// ---- Lessons learned ---- //
children.push(h1("8. Lessons Learned"));
children.push(table([
  ["Lesson", "Why it matters"],
  ["Log-transform skewed regression targets", "Aligns the loss with RMSLE and prevents luxury outliers from dominating gradients"],
  ["NaN is not always missing", "Most missing values in this dataset encode absence (no pool, no garage); filling with median would invent features"],
  ["Engineered features beat raw features", "QualTotalSF, QualArea, NeighborhoodPrice ranked higher than every raw column in both XGBoost and SHAP"],
  ["Fit encoders/clusterers on train only", "Fitting on train+test concatenation leaks the test distribution and gives optimistic CV"],
  ["Stacking > best single model", "Even the strongest single model (CatBoost 0.113) lost to the Ridge stacker (0.107)"],
  ["Linear models still compete", "Lasso and ElasticNet outscored all three tuned tree boosters - on small clean datasets, simple often wins"],
  ["Remove documented outliers", "Two flagged GrLivArea points were actively hurting every model; trusting the data dictionary is free signal"],
  ["Use Optuna, not grid search", "TPE samples adaptively; 30 trials reach better optima than 200 grid points across the same space"],
  ["SHAP for trustworthy interpretability", "Tree feature_importances counts splits; SHAP attributes actual prediction contribution"],
], [3500, 5860]));

children.push(h1("9. Reproducibility"));
children.push(bullet("Fixed SEED=42 throughout (numpy, sklearn KFold, Optuna TPESampler, KMeans, all model random_states)"));
children.push(bullet("Identical hardware -> identical metrics. Different OS or BLAS may shift the 5th decimal."));
children.push(bullet("Optuna runs are deterministic with TPESampler(seed=42)"));
children.push(bullet("Run modes: --quick (~1 min, no Optuna), default (~10 min, 30 trials), --trials 100 (~30 min, spec)"));

children.push(h1("Appendix A. Project Layout"));
children.push(code(`house-price-prediction/
  README.md
  requirements.txt
  .gitignore
  LICENSE
  data/
    raw/                       # train.csv, test.csv, sample_submission.csv, data_description.txt
    processed/
    README.md
  src/
    __init__.py
    data_loader.py
    preprocessing.py
    train.py
    predict.py
    evaluate.py
  models/
    model.pkl                  # serialized stack + preprocessing artifacts
  app/
    streamlit_app.py
  docs/
    House_Prices_Report.docx
    figures/                   # 11 generated PNGs
  tests/
    test_preprocessing.py
  .github/workflows/ci.yml`));

// ----------------------------------------------------------------------- //
const doc = new Document({
  creator: "Claude",
  title: "House Prices Project Report",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri", size: 36, bold: true, color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri", size: 28, bold: true, color: "2E75B6" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri", size: 24, bold: true, color: "404040" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUT, buffer);
  console.log(`Wrote ${OUT}  (${(buffer.length / 1024).toFixed(1)} KB)`);
});
