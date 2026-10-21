# ISBI documentation and reference audit — 2026-10-21

Scope: repository master at d4ad2c30170e039b3520c151e743e3efb89d0ccd,
the supplied SlotMIL_ISBI_Paper.zip, and the supplied rewritten manuscript.

## Changes

- README: dataset access and citation links; configuration sources and training
  example; arm results; spatial-reference results; original/amended H7 disclosure.
- Corrected README reference labeling: in-plane 0.8302, separable 0.8543.
- Removed universal-floor/ceiling language and unsupported equivalence implications.
- Merged four entries present in the uploaded bibliography but absent from GitHub:
  Jain/Wallace, Wiegreffe/Pinter, the LIDC data DOI, and MedMNIST v2.
- Added Chang et al. (implicit differentiation) and Loshchilov/Hutter (AdamW).
- In the updated upload package, cited these implementation sources, Bylinskii's
  saliency-metric analysis, and adjacent Slot-MIL/INSIGHT work in both supplied
  manuscript variants. Both variants and the original main-document selection
  are preserved.

## Checks and limits

All citation keys in repository paper TeX files and both supplied manuscript
variants resolve to bibliography entries; there are no duplicate keys. Both
updated uploaded variants build with pdfLaTeX/BibTeX to five pages, without
undefined citations or overfull-box warnings. Page count alone does not establish
conference formatting compliance. No model training or experiments were rerun.

Primary-source metadata was checked for Normal Guidance, Slot-Mixup/Slot-MIL,
INSIGHT, implicit differentiation, AdamW, MosMedData, both attention/explanation
papers, and the LIDC dataset citation. Core model and saliency source links were
also checked. Publisher access to three DOI landing pages returned HTTP 403;
these links were retained rather than classified as nonexistent. The official
MosMed portal could not be reached; its URL matches the dataset authors' paper.
Dataset acquisition remains subject to provider access terms.

The GitHub paper/main.tex is an older manuscript than the supplied package. This
PR does not silently replace that manuscript or its companion with the newer
upload. The README follows the revised uploaded manuscript; some historical
RESULTS.md and paper prose still require synchronization if this repository is
intended to host the exact final submission.

Bibliographic completeness is scoped to the described methods and supplied
claims, not a claim of an exhaustive literature review. Existing entries that
are not cited by a particular main manuscript are retained for companion and
historical analyses. BibTeX will include only cited entries in each document.
