# COV19-CT-DB access — GRANTED (2026-08-11)

Kollias replied: nothing further needed. Mohammad previously completed the EULA
and access persists under those existing terms.

**Still outstanding: the data itself.** The reply grants permission, not files.
COV19-CT-DB is distributed as download links (historically Dropbox) issued after
the EULA, and nothing matching `*cov19*` / `*covid*` exists on scratch — the old
`covid-challenge/` directory was purged along with the rest.

Fastest route, in order:
1. **Ask Mohammad for the download links** (or a copy of the archives if he still
   has them). He holds the EULA, so he almost certainly has the links, and this
   avoids another round-trip with Kollias.
2. If Mohammad no longer has them, reply to Kollias asking him to re-send the
   download links, referencing that access was confirmed on 2026-08-11.

Storage is no longer a blocker: 75 GB free vs ~45 GB needed (~26 GB raw
+ ~19 GB DINOv2 features). No need to touch the Lategame checkpoints.

Once the archives land, unpack to `data/cov19d/` and extract features with the
same pipeline as MosMed (`scripts/extract_mosmed.py` is the closest template —
COV19D ships JPEG slice folders per patient rather than NIfTI, so the loader
needs a directory-walking reader instead of nibabel).

---

## Original request (kept for reference)

Draft email for the COV19D / COV19-CT-DB dataset (plan.md line 70, needed ~W10
for the cross-domain generalisation experiment).

## Send to

**To:** d.kollias@qmul.ac.uk
**Subject:** COV19-CT-DB dataset request

## Two hard requirements

1. **Send from your institutional address** (`bthapar@terpmail.umd.edu`). The
   data is explicitly not released to personal email addresses, so a gmail
   sender will simply be dropped.
2. **Include your title and an official academic web page.** A UMD directory
   entry, lab page, or department profile all work. A personal site on a
   non-university domain generally does not.

## Draft

> Dear Dr Kollias,
>
> I am writing to request access to the COV19-CT-DB database for academic
> research.
>
> I am a [**FILL: e.g. senior undergraduate / MS student / PhD student**] in
> [**FILL: department**] at the University of Maryland, College Park, working
> with [**FILL: advisor name and title**]. My academic page is
> [**FILL: URL**].
>
> Our group previously worked with COV19-CT-DB through the MIA-COV19D
> competition, where we developed a gated-attention multiple-instance-learning
> ensemble over 3-D chest CT and reached a macro F1 of 0.928 on the challenge
> test set.
>
> I would now like to use the database for a new project on interpretable MIL
> pooling for volumetric CT. Specifically, we are studying whether slot
> attention, used as the pooling bottleneck in attention-based MIL, produces
> latent slots that align with distinct radiological findings. COV19-CT-DB's
> multi-centre structure is important to us as a cross-centre generalisation
> benchmark, complementing the lesion-mask datasets (LIDC-IDRI, MosMedData) we
> use for localisation evaluation. We intend to submit this work to ISBI/MICCAI.
>
> I will of course abide by the terms of use, restrict the data to academic
> non-commercial research, refrain from redistributing it, and cite the
> COV19-CT-DB and MIA-COV19D papers in any resulting publication.
>
> Thank you very much for your time and for making this resource available.
>
> Best regards,
> Bhavesh Thapar
> [**FILL: title**], [**FILL: department**]
> University of Maryland, College Park
> bthapar@terpmail.umd.edu

## Notes

- The prior-participation paragraph is the strongest part of this request —
  keep it, and correct the numbers if the final challenge figure differed. The
  surviving write-up records macro F1 = 0.9279 from a five-model ensemble.
  (plan.md line 25 says "seven Gated-Attention MIL models"; the write-up says
  five. Check which is right before sending — an inflated claim to the person
  who ran the competition is the wrong first impression.)
- Access is human-gated, so send early even though the data is not needed until
  W10. A follow-up after ~2 weeks of silence is reasonable.
- Ask your advisor to CC or send in parallel if a faculty sender would carry
  more weight.
- Storage is a separate decision: COV19D is ~26 GB raw plus ~19 GB of features,
  which does not fit alongside the current LIDC + MosMed budget without either
  freeing the 47 GB of Lategame checkpoints or requesting a quota increase.
