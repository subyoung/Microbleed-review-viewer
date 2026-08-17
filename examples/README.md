# Input format

The viewer takes two things: a findings workbook and a folder of MRI cases.
Nothing else is read, and neither is ever written to.

**The names below are only the defaults.** Sheet name, column names and
filename endings are all settable in `config.json`, and the dataset dialog can
read the endings off your data for you — see the README. This page describes
the shape, not a requirement.

`example_findings.xlsx` in this folder is a working example — invented subject
IDs and invented coordinates, but the real column names and the real sheet
name, so it imports.

## The findings workbook

One sheet named **`MCH-microhemorrage`**, one row per candidate microbleed,
with a header row first.

| Column | Required | Meaning |
| --- | :---: | --- |
| `subjectid` | ✔ | Case ID. Must match the folder name under the data root. |
| `RAS L-R L` | ✔ | Physical RAS coordinate of the finding, in millimetres. |
| `RAS P-A A` | ✔ | |
| `RAS I-S S` | ✔ | |
| `verify (yes=1)` | ✔ | The source verdict: `1` yes, `0` no, blank not recorded. |
| `slicecount` | | Slice the finding was found on, for reference only. |
| `atlasregions` | | Anatomical region, shown beside the finding. |
| `RAS verified` | | Whether the source coordinate was checked. |
| `2DQSM`, `SWI`, `3DQSM` | | Per-sequence source flags, shown in the report. |
| `readers` | | Who recorded it in the source. |
| `need adjudicate` | | Free text; a non-empty value shows as "needs adjudication". |
| `comments` | | Free text from the source. |

A row with no `subjectid` is skipped. A row whose three RAS values are not
numbers stops the import and names the Excel row, rather than importing a
finding at the wrong place.

The coordinates are **physical RAS in millimetres**, not voxel indices. The
viewer maps them through each image's own affine, so they stay correct
whichever way the volumes are stored on disk.

## The data root

One folder per case, named exactly as `subjectid`:

```text
Data/
├── SUBJ0001/
│   ├── SUBJ0001-20240115_090000_GRE4_SWI_AffineRestored.nii.gz
│   ├── SUBJ0001-20240115_090000_GRE4_SWI_axiMIP2_AffineRestored.nii.gz
│   └── SUBJ0001-20240115_090000_GRE4_chi_nSFCR+0_Avg_AffineRestored.nii.gz
└── SUBJ0002/
    └── …
```

Three sequences are recognised, by **suffix**:

| Shown as | Filename must end with |
| --- | --- |
| SWI | `_GRE4_SWI_AffineRestored.nii.gz` |
| SWI MIP | `_GRE4_SWI_axiMIP2_AffineRestored.nii.gz` |
| QSM | `_chi_nSFCR+0_Avg_AffineRestored.nii.gz` |

Anything before the suffix is free. A case may have one, two or all three; a
missing sequence is reported in place as *Not available* rather than
substituted with another file, because a mask drawn on the wrong image is
worse than no image.

**Only `AffineRestored` products are accepted.** The three are expected to
share a voxel grid — that is what lets one segmentation belong to the case
rather than to a sequence — and the viewer says so plainly when they do not.

## What the viewer writes

Never the workbook, and never the MRI files. Everything a reader produces goes
to two places, both configurable:

```text
microbleed_review_data.sqlite          verdicts, comments, positions, sessions, log
labels/<reader>/<case>_round<n>.nii.gz one file per case and reader, all its masks
```

A label file is written in the source image's own voxel order and affine, with
both `sform` and `qform` set, so it opens aligned in any other viewer. Each
finding has its own integer; the `segmentations` sheet of the export says which
integer is which finding.
