AMP review candidate — 12 October 2026

STATUS: Complete for substantive review; NO_GO for journal submission.
Candidate: Bioinformatics Original Paper, format-free initial-review layout.
This generic article.cls source is not the official production template.

Overleaf: upload this directory as a ZIP. Select pdfLaTeX and main.tex.
References use BibTeX/plainnat. Compile supplement.tex separately for the
single combined supplementary PDF; cover_letter.tex and decision.tex are
separate administrative review documents. Never send the cover-letter draft
until human approval and scientific blockers have been resolved.

Local build (TeX Live with standard packages):
  latexmk -pdf main.tex
  latexmk -pdf supplement.tex
  latexmk -pdf cover_letter.tex
  latexmk -pdf decision.tex

The figure PDF is included. Its editable deterministic source is
figures/reconstruction_generation.tex. Compile it with pdfLaTeX if editing.
Do not interpret a successful TeX build as scientific or submission approval.

The evidence directory in the full review bundle holds summaries and hashes.
It is not a full model-reproduction release: inputs, checkpoints and weights
are not part of the Overleaf source package. Historical 27-peptide tables
are independent historical records, not validation of H5.

Unconfirmed declarations: author details and consent, contributions, funding,
competing interests, overlapping resources with related work, complete AI
history and data/code release arrangement. No human signoff is asserted.
