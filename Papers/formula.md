# NeurIPS 2026 Formatting Reference

## Page Limits
- **Content**: up to **9 pages** (including figures)
- **Not counted**: acknowledgments, references, checklist, appendices
- **Appendix**: no page limit, but paper must stand alone without it
- **Paper size**: US Letter

## Submission Rules
- Default option = Main Track, double-blind review
- Do NOT use `final` option until accepted
- Do NOT modify style file parameters
- Do NOT refer to line numbers in submission
- `ack` environment auto-hides in anonymized submission
- Refer to own work in third person (double-blind)
- Checklist is **mandatory** — desk reject without it

## Tracks
| Track | Option |
|---|---|
| Main (default) | `main` |
| Position Paper | `position` |
| Evaluations & Datasets | `eandd` |
| Creative AI | `creativeai` |
| Workshop (single-blind) | `sglblindworkshop` |
| Workshop (double-blind) | `dblblindworkshop` |

Camera-ready: add `final`, e.g. `\usepackage[main, final]{sty/neurips_2026}`

## Typography Quick Reference
- Body: 10pt, Times New Roman, 11pt leading
- Title: 17pt bold centered
- Section: 12pt bold flush left, lower case
- Subsection/Subsubsection: 10pt bold flush left
- No paragraph indentation, 5.5pt paragraph spacing
- Math: use `\[...\]`, `equation`, `align` — never `$$...$$`

## Figures & Tables
- Figure caption **after** figure
- Table caption **before** table
- Use `booktabs`, no vertical rules
- `\includegraphics[width=0.8\linewidth]{file.pdf}`

## Citations
- `natbib` loaded by default; `\citet{}` inline, `\citep{}` parenthetical
- Author: `\And` (auto line break), `\AND` (forced line break)
- `\thanks{}` for author footnotes (not funding)

## Checklist Macros
- `\answerYes{}`, `\answerNo{}`, `\answerNA{}`, `\answerTODO{}`

## Fonts
- PDF must contain only Type 1 or Embedded TrueType fonts
- Use `pdflatex` to compile
- Use `amsfonts` (not `\bbold`)
