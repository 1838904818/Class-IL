const pptxgen = require('pptxgenjs');
const fs = require('fs');
const path = require('path');

const C = {
  navy: '0B1F33',
  ink: '172431',
  slate: '5F6E7C',
  mist: 'EAF0F4',
  pale: 'F6F8FA',
  white: 'FFFFFF',
  teal: '0C7C86',
  tealPale: 'DDF0F1',
  berry: '7C3F68',
  berryPale: 'F1E5ED',
  green: '356B4D',
  greenPale: 'E4EFE8',
  gold: 'B47B18',
  goldPale: 'F7EDD8',
  red: '9B3B3B',
  redPale: 'F7E7E7',
  grid: 'CBD5DE',
};

const W = 13.333;
const H = 7.5;
const ROOT = path.resolve(__dirname, '..', '..');
const FIG = path.join(ROOT, 'paper_ncd', 'figures');

function parseArgs(argv) {
  const out = { output: path.join(__dirname, 'NCD_Evidence_Update_2026-10-10.pptx'), notes: null };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--output') out.output = path.resolve(argv[++i]);
    else if (argv[i] === '--notes') out.notes = path.resolve(argv[++i]);
    else throw new Error(`Unknown argument: ${argv[i]}`);
  }
  return out;
}

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: opts.fontFace || 'Arial',
    fontSize: opts.fontSize || 18,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    italic: opts.italic || false,
    align: opts.align || 'left',
    valign: opts.valign || 'mid',
    margin: opts.margin === undefined ? 0 : opts.margin,
    breakLine: false,
    fit: opts.fit || undefined,
  });
}

function addTitle(slide, title, subtitle = '') {
  addText(slide, title, 0.65, 0.38, 11.95, 0.48, {
    fontFace: 'Cambria', fontSize: 30, bold: true, color: C.navy,
  });
  if (subtitle) addText(slide, subtitle, 0.67, 0.90, 11.8, 0.30, { fontSize: 11.5, color: C.slate });
}

function addFooter(slide, source, slideNo, dark = false) {
  const color = dark ? 'D5DEE7' : C.slate;
  addText(slide, source, 0.65, 7.18, 11.25, 0.18, { fontSize: 8.5, color });
  addText(slide, String(slideNo), 12.15, 7.16, 0.50, 0.20, { fontSize: 9, color, align: 'right' });
}

function addPill(slide, text, x, y, w, fill, color = C.white) {
  slide.addShape('roundRect', { x, y, w, h: 0.34, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
  addText(slide, text, x + 0.08, y, w - 0.16, 0.34, { fontSize: 10, bold: true, color, align: 'center' });
}

function addCard(slide, x, y, w, h, opts = {}) {
  slide.addShape('roundRect', {
    x, y, w, h,
    rectRadius: 0.08,
    fill: { color: opts.fill || C.white },
    line: { color: opts.line || C.grid, width: opts.lineWidth || 1 },
    shadow: opts.shadow === false ? undefined : { type: 'outer', color: '7D8B98', blur: 1.2, angle: 45, distance: 1, opacity: 0.16 },
  });
}

function addCallout(slide, big, label, x, y, w, fill, color = C.navy) {
  addCard(slide, x, y, w, 1.22, { fill, line: fill, shadow: false });
  addText(slide, big, x + 0.15, y + 0.10, w - 0.30, 0.62, { fontFace: 'Cambria', fontSize: 34, bold: true, color, align: 'center', fit: 'shrink' });
  addText(slide, label, x + 0.16, y + 0.73, w - 0.32, 0.32, { fontSize: 11.5, color, align: 'center', fit: 'shrink' });
}

function addTable(slide, x, y, widths, heights, rows, opts = {}) {
  let yy = y;
  rows.forEach((row, r) => {
    let xx = x;
    row.forEach((cell, c) => {
      const isHead = r === 0;
      const fill = cell.fill || (isHead ? (opts.headerFill || C.navy) : (r % 2 === 0 ? C.pale : C.white));
      const color = cell.color || (isHead ? C.white : C.ink);
      slide.addShape('rect', {
        x: xx, y: yy, w: widths[c], h: heights[r],
        fill: { color: fill },
        line: { color: opts.lineColor || C.grid, width: 0.8 },
      });
      addText(slide, cell.text, xx + 0.06, yy + 0.02, widths[c] - 0.12, heights[r] - 0.04, {
        fontSize: cell.fontSize || (isHead ? 10.5 : 10.5),
        bold: cell.bold === undefined ? isHead : cell.bold,
        color,
        align: cell.align || (c < 2 ? 'left' : 'center'),
      });
      xx += widths[c];
    });
    yy += heights[r];
  });
}

function addNotes(slide, notes, index) {
  if (notes && notes[index]) slide.addNotes(notes[index]);
}

function buildDeck(notes = []) {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_WIDE';
  pptx.author = 'OFRA research project';
  pptx.subject = 'Noise-calibrated silent explanation drift evidence update';
  pptx.title = 'NCD Evidence Update — Architecture and Results';
  pptx.company = 'Universiti Malaya research project';
  pptx.lang = 'en-US';
  pptx.theme = {
    headFontFace: 'Cambria',
    bodyFontFace: 'Arial',
    lang: 'en-US',
  };

  // Slide 1 — title and evidence boundary.
  {
    const s = pptx.addSlide();
    s.background = { color: C.navy };
    addPill(s, 'INTERIM EVIDENCE · 10 OCT 2026', 0.72, 0.55, 2.80, C.teal);
    addText(s, 'Noise-Calibrated\nSilent Explanation Drift', 0.75, 1.22, 7.10, 1.55, {
      fontFace: 'Cambria', fontSize: 42, bold: true, color: C.white, valign: 'top',
    });
    addText(s, 'Architecture, intervention benchmark, and partial real-data evidence', 0.78, 2.94, 7.35, 0.42, {
      fontSize: 17, color: 'D7E2EB',
    });
    addCard(s, 8.48, 1.12, 3.95, 4.80, { fill: '132D45', line: '26445D', shadow: false });
    addText(s, '20 / 140', 8.82, 1.72, 3.28, 0.92, { fontFace: 'Cambria', fontSize: 52, bold: true, color: C.white, align: 'center' });
    addText(s, 'registered Exp. 3 Phase-I runs\ncurrently in the local interim slice', 8.88, 2.72, 3.15, 0.78, { fontSize: 14, color: 'D7E2EB', align: 'center' });
    addPill(s, 'NOT THE FINAL FOUR-DATASET RESULT', 8.92, 3.69, 3.07, C.berry);
    addText(s, 'Current verdict', 8.93, 4.35, 3.02, 0.30, { fontSize: 12, bold: true, color: '96D5D9', align: 'center' });
    addText(s, 'Operable measurement; saturated decision; no established innovation.', 8.84, 4.76, 3.20, 0.74, { fontSize: 17, bold: true, color: C.white, align: 'center' });
    addText(s, 'Locked target: prediction · forgetting · silent explanation drift · non-suppressive governance', 0.78, 6.43, 11.45, 0.38, { fontSize: 13, color: 'D7E2EB' });
    addFooter(s, 'Evidence scope: frozen Exp. 2 + interim Exp. 3 subset; full HPC comparison pending', 1, true);
    addNotes(s, notes, 0);
  }

  // Slide 2 — locked research architecture.
  {
    const s = pptx.addSlide();
    s.background = { color: C.white };
    addTitle(s, 'Research architecture: questions before mechanisms', 'The research object is not NCD, OFRA, BARE, SHAP, or a learner swap.');
    addCard(s, 0.72, 1.35, 11.88, 1.04, { fill: C.navy, line: C.navy, shadow: false });
    addText(s, 'Can a task-free class-incremental NIDS preserve useful predictions while its explanation basis changes silently?', 1.04, 1.55, 11.20, 0.60, { fontFace: 'Cambria', fontSize: 23, bold: true, color: C.white, align: 'center' });

    const cards = [
      { x: 0.72, tag: 'RQ1 · MANDATORY', fill: C.tealPale, line: C.teal, title: 'Measure silent drift', body: 'Frequency, magnitude, agreement, and threshold sensitivity across checkpoints.' },
      { x: 4.47, tag: 'RQ2 · MATCHED', fill: C.greenPale, line: C.green, title: 'Quantify trade-offs', body: 'Accuracy, class safety, forgetting, memory, runtime, and explanation cost.' },
      { x: 8.22, tag: 'RQ3 · OPEN', fill: C.redPale, line: C.red, title: 'Test governance', body: 'Non-suppressive evidence logging with binary decision invariance. Not yet validated.' },
    ];
    cards.forEach((c) => {
      addCard(s, c.x, 2.82, 3.45, 2.82, { fill: c.fill, line: c.line });
      addPill(s, c.tag, c.x + 0.28, 3.08, 1.62, c.line);
      addText(s, c.title, c.x + 0.28, 3.63, 2.90, 0.48, { fontFace: 'Cambria', fontSize: 21, bold: true, color: C.navy });
      addText(s, c.body, c.x + 0.28, 4.25, 2.88, 0.92, { fontSize: 14, color: C.ink, valign: 'top' });
    });
    addCard(s, 1.65, 6.05, 10.02, 0.68, { fill: C.goldPale, line: C.gold, shadow: false });
    addText(s, 'NCD is a candidate measurement mechanism. A detector gain, threshold, wrapper, or model swap alone cannot satisfy the substantive-innovation gate.', 1.95, 6.18, 9.42, 0.36, { fontSize: 13.5, bold: true, color: C.navy, align: 'center' });
    addFooter(s, 'Source: RESEARCH_TARGET_LOCK.md and paper_ncd/paper.tex', 2);
    addNotes(s, notes, 1);
  }

  // Slide 3 — full architecture figure.
  {
    const s = pptx.addSlide();
    s.background = { color: C.pale };
    s.addImage({ path: path.join(FIG, 'ncd_method_architecture.png'), x: 1.39, y: 0.36, w: 10.54, h: 6.86 });
    addFooter(s, 'Frozen observations → repeated explanations → permutation/FDR → 2-of-3 consensus → non-suppressive audit', 3);
    addNotes(s, notes, 2);
  }

  // Slide 4 — intervention benchmark counts.
  {
    const s = pptx.addSlide();
    s.background = { color: C.white };
    addTitle(s, 'Exp. 2: intervention-grounded detector counts', 'Five seeds per cell; B2 is the repeat-mean Jaccard control.');
    const rows = [
      ['Dataset', 'Learner', 'NCD FP', 'B2 FP', 'NCD P1', 'B2 P1', 'NCD P2', 'B2 P2'],
      ['NSL-KDD', 'Replay', '0/34', '0/34', '8/8', '8/8', '5/5', '5/5'],
      ['UNSW-NB15', 'Replay', '0/30', '0/30', '5/5', '5/5', '4/4', '4/4'],
      ['NSL-KDD', 'iCaRL', '0/20', '0/20', '9/9', '9/9', '—', '—'],
      ['UNSW-NB15', 'iCaRL', '1/20', '0/20', '4/4', '4/4', '—', '—'],
    ].map((row, ri) => row.map((text, ci) => ({
      text,
      fill: ri === 4 && ci === 2 ? C.redPale : undefined,
      color: ri === 4 && ci === 2 ? C.red : undefined,
      bold: ri === 4 && ci === 2 ? true : undefined,
    })));
    addTable(s, 0.72, 1.48, [1.70, 1.30, 1.48, 1.42, 1.48, 1.42, 1.48, 1.42], [0.58, 0.62, 0.62, 0.62, 0.62], rows);
    addCallout(s, '100%', 'eligible P1/P2 cases detected by both NCD and B2', 0.82, 5.15, 3.55, C.tealPale, C.teal);
    addCallout(s, '1 vs 0', 'false positives: NCD versus B2 in UNSW iCaRL', 4.88, 5.15, 3.55, C.redPale, C.red);
    addCallout(s, '20–34', 'negative controls per cell: too few for population FPR <5%', 8.94, 5.15, 3.55, C.goldPale, C.gold);
    addCard(s, 1.63, 6.55, 10.05, 0.46, { fill: C.navy, line: C.navy, shadow: false });
    addText(s, 'Decision: benchmark operability is supported; NCD superiority is not.', 1.85, 6.60, 9.60, 0.30, { fontSize: 16, bold: true, color: C.white, align: 'center' });
    addFooter(s, 'Source: frozen Exp. 2 five-seed intervention matrix', 4);
    addNotes(s, notes, 3);
  }

  // Slide 5 — ROC figure.
  {
    const s = pptx.addSlide();
    s.background = { color: C.white };
    addTitle(s, 'Exp. 2: same-slice separability is perfect—and non-identifying', 'All six continuous scores reach AUROC 1.000 in each frozen dataset–learner cell.');
    s.addImage({ path: path.join(FIG, 'exp2_four_cell_roc.png'), x: 0.67, y: 1.32, w: 7.62, h: 5.56 });
    addCard(s, 8.66, 1.58, 3.90, 4.78, { fill: C.pale, line: C.grid });
    addText(s, 'What the curves show', 9.00, 1.92, 3.22, 0.42, { fontFace: 'Cambria', fontSize: 21, bold: true, color: C.navy });
    addText(s, 'Constructed eligible positives are separable from negative controls.', 9.00, 2.52, 3.20, 0.74, { fontSize: 15, color: C.ink, valign: 'top' });
    addText(s, 'What they do not show', 9.00, 3.45, 3.22, 0.42, { fontFace: 'Cambria', fontSize: 21, bold: true, color: C.berry });
    addText(s, 'External calibration, deployment discrimination, causal validity, or NCD superiority.', 9.00, 4.05, 3.20, 0.94, { fontSize: 15, color: C.ink, valign: 'top' });
    addPill(s, 'NO NOVELTY CLAIM', 9.28, 5.42, 2.62, C.berry);
    addFooter(s, 'Source: paper_ncd/figures/exp2_four_cell_roc.png', 5);
    addNotes(s, notes, 4);
  }

  // Slide 6 — partial real-data metrics.
  {
    const s = pptx.addSlide();
    s.background = { color: C.white };
    addTitle(s, 'Exp. 3 interim: performance and drift rates', '20/140 Phase-I runs: two datasets × Replay/iCaRL × five seeds. Percentages are seed means or pooled eligible-unit rates.');
    const rows = [
      ['Dataset', 'Learner', 'Acc.', 'BAcc.', 'Macro F1', 'Forget.', 'B1 drift', 'B2 drift', 'NCD drift'],
      ['NSL-KDD', 'Replay', '79.98', '59.96', '59.73', '2.82', '62.96', '59.26', '100.00'],
      ['NSL-KDD', 'iCaRL', '77.54', '70.90', '62.68', '6.57', '30.30', '30.30', '96.97'],
      ['UNSW-NB15', 'Replay', '63.30', '42.13', '34.00', '33.44', '31.67', '30.00', '98.33'],
      ['UNSW-NB15', 'iCaRL', '51.22', '46.06', '28.15', '35.45', '18.87', '18.87', '98.11'],
    ].map((row, ri) => row.map((text, ci) => ({
      text,
      fill: ri > 0 && ci === 8 ? C.redPale : undefined,
      color: ri > 0 && ci === 8 ? C.red : undefined,
      bold: ri > 0 && ci === 8 ? true : undefined,
    })));
    addTable(s, 0.62, 1.50, [1.70, 1.20, 1.25, 1.25, 1.35, 1.20, 1.30, 1.30, 1.40], [0.58, 0.64, 0.64, 0.64, 0.64], rows);
    addCard(s, 0.86, 5.28, 3.48, 1.18, { fill: C.greenPale, line: C.green, shadow: false });
    addText(s, '2.82–35.45%', 1.05, 5.48, 3.10, 0.42, { fontFace: 'Cambria', fontSize: 28, bold: true, color: C.green, align: 'center' });
    addText(s, 'mean forgetting across cells', 1.08, 5.94, 3.04, 0.24, { fontSize: 11.5, color: C.ink, align: 'center' });
    addCard(s, 4.92, 5.28, 3.48, 1.18, { fill: C.redPale, line: C.red, shadow: false });
    addText(s, '96.97–100%', 5.10, 5.48, 3.12, 0.42, { fontFace: 'Cambria', fontSize: 28, bold: true, color: C.red, align: 'center' });
    addText(s, 'NCD drift rate: near saturation', 5.11, 5.94, 3.10, 0.24, { fontSize: 11.5, color: C.ink, align: 'center' });
    addCard(s, 8.98, 5.28, 3.48, 1.18, { fill: C.goldPale, line: C.gold, shadow: false });
    addText(s, '20 / 140', 9.18, 5.48, 3.08, 0.42, { fontFace: 'Cambria', fontSize: 28, bold: true, color: C.gold, align: 'center' });
    addText(s, 'registered runs represented', 9.17, 5.94, 3.10, 0.24, { fontSize: 11.5, color: C.ink, align: 'center' });
    addFooter(s, 'Interim only: excludes CIC-IDS2017, CIC-IDS2018, and five learner families', 6);
    addNotes(s, notes, 5);
  }

  // Slide 7 — confusion matrices.
  {
    const s = pptx.addSlide();
    s.background = { color: C.white };
    addTitle(s, 'Final-checkpoint confusion matrices expose class-specific failure', 'Raw counts pooled over seeds 1, 2, 3, 4, and 42, then row-normalized within each true class.');
    s.addImage({ path: path.join(FIG, 'partial_exp3_confusion_matrices.png'), x: 0.58, y: 1.30, w: 8.54, h: 5.72 });
    addCard(s, 9.44, 1.60, 3.17, 4.72, { fill: C.pale, line: C.grid });
    addText(s, 'Reading rule', 9.75, 1.93, 2.54, 0.38, { fontFace: 'Cambria', fontSize: 20, bold: true, color: C.navy });
    addText(s, 'Diagonal cells are per-class recall. Off-diagonal cells show which attacks collapse into another class or Normal.', 9.75, 2.47, 2.54, 1.08, { fontSize: 14.5, color: C.ink, valign: 'top' });
    addText(s, 'Why it matters', 9.75, 3.75, 2.54, 0.38, { fontFace: 'Cambria', fontSize: 20, bold: true, color: C.teal });
    addText(s, 'Headline accuracy cannot certify class safety. The full result must retain per-class and Normal behaviour.', 9.75, 4.29, 2.54, 1.05, { fontSize: 14.5, color: C.ink, valign: 'top' });
    addPill(s, 'DESCRIPTIVE · 20 RUNS', 9.80, 5.60, 2.45, C.gold);
    addFooter(s, 'Source: paper_ncd/figures/partial_exp3_confusion_matrices.png', 7);
    addNotes(s, notes, 6);
  }

  // Slide 8 — threshold sensitivity and cost.
  {
    const s = pptx.addSlide();
    s.background = { color: C.white };
    addTitle(s, 'Threshold audit: calibration does not prevent saturation', 'Pooled silent units at the five-point recall tolerance.');
    s.addImage({ path: path.join(FIG, 'partial_exp3_threshold_sensitivity.png'), x: 0.62, y: 1.30, w: 8.36, h: 3.09 });
    addCard(s, 9.30, 1.42, 3.28, 2.74, { fill: C.redPale, line: C.red, shadow: false });
    addText(s, '93.94–100%', 9.58, 1.80, 2.72, 0.62, { fontFace: 'Cambria', fontSize: 32, bold: true, color: C.red, align: 'center' });
    addText(s, 'NCD drift across q = 0.01–0.10', 9.62, 2.48, 2.64, 0.46, { fontSize: 13, color: C.ink, align: 'center' });
    addText(s, 'Flatness here is an undesirable saturation result—not threshold robustness.', 9.60, 3.12, 2.68, 0.68, { fontSize: 14, bold: true, color: C.red, align: 'center' });
    addCallout(s, '93.94–100%', 'all-explainer agreement; invalidated by decision saturation', 0.74, 4.90, 3.52, C.tealPale, C.teal);
    addCallout(s, '46.99–106.86 s', 'NCD permutation time per dataset–method cell over five seeds', 4.90, 4.90, 3.52, C.goldPale, C.gold);
    addCallout(s, '3569–4057×', 'fixed-rule decision time, despite only 3.29–3.65% of explanation time', 9.06, 4.90, 3.52, C.berryPale, C.berry);
    addCard(s, 1.62, 6.55, 10.08, 0.46, { fill: C.navy, line: C.navy, shadow: false });
    addText(s, 'Decision: the calibrated rule is currently less selective and more expensive than the matched fixed controls.', 1.86, 6.60, 9.60, 0.30, { fontSize: 15, bold: true, color: C.white, align: 'center' });
    addFooter(s, 'Source: q/Jaccard sensitivity and runtime aggregation for the 20-run subset', 8);
    addNotes(s, notes, 7);
  }

  // Slide 9 — completed NSL-KDD drift-only slice.
  {
    const s = pptx.addSlide();
    s.background = { color: C.white };
    addTitle(s, 'NSL-KDD 25-run drift slice: saturation persists', 'Five learners × five seeds; matched performance aggregation remains queued.');
    s.addImage({ path: path.join(FIG, 'nsl_25run_drift_saturation.png'), x: 0.48, y: 1.28, w: 8.72, h: 4.82 });
    addCard(s, 9.48, 1.42, 3.04, 4.64, { fill: C.pale, line: C.grid });
    addText(s, '154 / 157', 9.78, 1.82, 2.44, 0.62, { fontFace: 'Cambria', fontSize: 33, bold: true, color: C.red, align: 'center' });
    addText(s, 'silent units marked by NCD', 9.82, 2.47, 2.36, 0.36, { fontSize: 12.5, color: C.ink, align: 'center' });
    addText(s, '91.72%', 9.82, 3.10, 2.36, 0.44, { fontFace: 'Cambria', fontSize: 26, bold: true, color: C.teal, align: 'center' });
    addText(s, 'all-explainer agreement', 9.82, 3.56, 2.36, 0.28, { fontSize: 12, color: C.ink, align: 'center' });
    addText(s, 'B1 70.70% · B2 69.43%', 9.82, 3.88, 2.36, 0.28, { fontSize: 11.5, bold: true, color: C.slate, align: 'center' });
    addText(s, '68 / 68', 9.82, 4.42, 2.36, 0.44, { fontFace: 'Cambria', fontSize: 26, bold: true, color: C.gold, align: 'center' });
    addText(s, 'future-observed units are positive; no comparison group', 9.82, 4.90, 2.36, 0.56, { fontSize: 12, color: C.ink, align: 'center', valign: 'top' });
    addPill(s, 'AGREEMENT VIA SATURATION', 9.68, 5.60, 2.64, C.berry);
    addCard(s, 1.50, 6.40, 10.36, 0.52, { fill: C.navy, line: C.navy, shadow: false });
    addText(s, 'Finding: a completed larger slice confirms saturation; it does not establish prediction, causality, governance, or innovation.', 1.75, 6.48, 9.86, 0.30, { fontSize: 14.5, bold: true, color: C.white, align: 'center' });
    addFooter(s, 'Source: DICC jobs 483664 and 483686; protected 25-run NSL-KDD drift evidence', 9);
    addNotes(s, notes, 8);
  }

  // Slide 10 — bounded conclusion and current execution.
  {
    const s = pptx.addSlide();
    s.background = { color: C.navy };
    addPill(s, 'CURRENT EVIDENCE VERDICT', 0.72, 0.50, 2.60, C.teal);
    addText(s, 'The measurement works.\nThe innovation claim does not—yet.', 0.78, 1.18, 7.35, 1.28, { fontFace: 'Cambria', fontSize: 36, bold: true, color: C.white, valign: 'top' });
    const items = [
      ['SUPPORTED', 'Repeated explanations and intervention constructions are auditable.'],
      ['NEGATIVE', 'NCD ties B2 on eligible positives, adds one false positive, and marks 154/157 silent units in the completed NSL drift slice.'],
      ['OPEN', 'Matched NSL performance, the other three datasets, cross-explainer validity, future-recall relation, and RQ3 governance.'],
    ];
    items.forEach((it, i) => {
      const y = 2.92 + i * 1.03;
      const fill = i === 0 ? C.green : (i === 1 ? C.berry : C.gold);
      addPill(s, it[0], 0.82, y, 1.28, fill);
      addText(s, it[1], 2.32, y - 0.03, 6.60, 0.46, { fontSize: 15, color: C.white });
    });
    addCard(s, 9.20, 1.18, 3.20, 4.96, { fill: '132D45', line: '26445D', shadow: false });
    addText(s, 'HPC completion path', 9.52, 1.55, 2.55, 0.44, { fontFace: 'Cambria', fontSize: 21, bold: true, color: C.white, align: 'center' });
    addText(s, '12 registered owned jobs\n\n2 complete\n10 queued or dependency-held\n\n4 primary runs\n4 performance aggregations\n4 drift aggregations\n\nNo duplicate submission.', 9.54, 2.22, 2.50, 3.38, { fontSize: 14, color: 'D7E2EB', align: 'center', valign: 'top' });
    addText(s, 'No current result establishes substantive innovation, cross-domain generality, or validated governance.', 0.82, 6.40, 11.38, 0.48, { fontSize: 15, bold: true, color: 'F0C8DE' });
    addFooter(s, 'Next: audit matched NSL performance and each remaining terminal protected result before replacing interim tables', 10, true);
    addNotes(s, notes, 9);
  }

  return pptx;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  let notes = [];
  if (args.notes) {
    notes = JSON.parse(fs.readFileSync(args.notes, 'utf8'));
    if (!Array.isArray(notes) || notes.length !== 10) throw new Error('Speaker notes JSON must contain exactly 10 strings.');
  }
  fs.mkdirSync(path.dirname(args.output), { recursive: true });
  const pptx = buildDeck(notes);
  await pptx.writeFile({ fileName: args.output });
  console.log(`saved=${args.output}`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error.stack || String(error));
    process.exit(1);
  });
}

module.exports = { buildDeck };
