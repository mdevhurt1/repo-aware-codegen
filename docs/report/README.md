# Report

LaTeX source for the ENPM703 Group 2 report, set in the CVPR 2017 author-kit style (two columns, 10pt Times, IEEE-style references).

## Build

```bash
cd docs/report
latexmk -pdf report.tex   # writes report.pdf
latexmk -c                # removes the intermediate files
```

This needs `latexmk` and a TeX Live install; on Ubuntu the packages are `texlive-latex-recommended` and `texlive-fonts-recommended`. On Overleaf, upload this folder and set `report.tex` as the main document.

`report.pdf` and the intermediate files are gitignored. To commit a submitted copy, add it with `git add -f`.

## Files

| File | What |
|---|---|
| `report.tex` | The report. The preamble follows the kit's `egpaper_final.tex` |
| `references.bib` | Bibliography; cite with `\cite{key}` |
| `figures/` | Figures; `\includegraphics{name}` finds them here |
| `cvpr.sty`, `ieee.bst` | Page style and reference style, copied unmodified from the CVPR 2017 author kit. Do not edit |

## Conventions

- Every unwritten part is marked with `\todo{...}`, which prints in red. Delete each marker as you write its section; `grep -n '\\todo{' report.tex` lists what is left.
- Size a figure as a multiple of `\linewidth`, and use `figure*` or `table*` for one that spans both columns. The commented block in the Experiments section of `report.tex` shows the kit's figure and table patterns.
- When you add a reference, take the title, authors and year from the paper's own page, and wrap any capitalization that must survive in braces, as in `{RepoBench}`.
- `report.tex` builds the final-copy layout with named authors. Commenting out `\cvprfinalcopy` switches to the kit's anonymous review layout with line numbers.
