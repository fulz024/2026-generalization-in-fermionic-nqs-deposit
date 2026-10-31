# Data and code for

**What makes fermionic neural-network quantum states generalize beyond their training configurations?**

This archive contains the processed tables and plotting code for the current
five main-text figures and eight Supplemental Material figures. It also
includes the numerical control tables and input hashes listed in the manifest.
Reproduction starts from these tables, not from retraining. No training
checkpoints or external molecular Hamiltonians are required. The small
reference inputs include the Hubbard exact target vector used to compute the
ranked-prefix weights and the H10 exact reference energy.

## Quick start

The figure pipeline was tested with Python 3.10. Activate a Python environment,
then run from the archive root:

```text
python -m pip install -r requirements.txt
python scripts/reproduce_all.py
```

The last command regenerates the following PDFs under `paper/figures/main/`.
The filenames retain their research identifiers; figure numbers follow the
current manuscript, not the numeral embedded in each filename.

| Manuscript item | Output |
|---|---|
| Fig. 1: ansatz schematic | `fig0_ansatz_schematic_decoder.pdf` |
| Fig. 2: architecture comparison | `fig1_hubbard_generalization.pdf` |
| Fig. 3: Slater baseline and NNBF projection | `fig2_slater_reference_predictions.pdf` |
| Fig. 4: H10 orbital-basis comparison | `fig3_h10_reference.pdf` |
| Fig. 5: MPS ordering comparison | `fig5_mps_generalization.pdf` |
| Fig. S1: interaction strength | `figS_interaction_strength.pdf` |
| Fig. S2: Born-sampled supports | `figS_stochastic_born_support.pdf` |
| Fig. S3: decoder control | `figS_decoder_comparison.pdf` |
| Fig. S4: extended MPS ordering | `figS_extended_mps_ordering.pdf` |
| Fig. S5: selected molecular spaces | `fig3_selected_spaces.pdf` |
| Fig. S6: H10 truncation | `figS_h10_truncation_curve.pdf` |
| Fig. S7: off-sample VMC updates | `fig4_vmc_reorganized.pdf` |
| Fig. S8: common-never-sampled VMC bands | `figS_band_control.pdf` |

The output-to-script and input-table mapping, with SHA-256 hashes, is recorded
in `analysis/generalization_reproducibility_manifest.json`. Some layout scripts
produce several related figures; the runner checks all 13 manuscript outputs.
Use `python scripts/reproduce_all.py --record-inputs` to audit the table reads.

The Slater point in Fig. 3 is selected by training agreement from the 50
K=4096 orbital fits, with 144 trainable orbital entries. It numerically
coincides with the analytic U=0 determinant used for the projection direction.
Lattice backflow freezes those reference orbitals and fits only two backflow
coefficients. Legacy table columns retain names such as `C_hidden` (the current
unseen score `C_U`) and `visible_corrected_*`; the manifest explains their use.
The current Slater baseline reads its fit table rather than the legacy analytic
reference row. No new training or outcome-based checkpoint selection is done.

Provenance paths in the tables and summaries are relative to the original
research directory. Machine-specific prefixes and account names have been
removed. Paths to source checkpoints or configurations identify the original
runs; those files are not required or included in this processed-data archive.

## Layout

| Path | Contents |
|---|---|
| `analysis/` | Processed CSV, JSON, and TSV tables consumed by the figure scripts |
| `reference/` | Small reference inputs for target weights and the VMC energy plot |
| `scripts/` | Current plotting scripts, their local dependencies, and the runner |
| `paper/figures/main/` | Figure outputs, with research filenames retained |

The pinned dependencies in `requirements.txt` cover the table-to-figure
pipeline. The archive supports numerical table-to-figure reproduction; it
does not reproduce ED, DMRG, supervised training, or VMC optimization from
scratch. Those require the research training code, checkpoints, and benchmark
Hamiltonians, which are outside this processed-data archive. The SI specifies
the corresponding ansatzes, optimization protocols, and evaluation conventions.

## Licensing and citation

Code is released under the MIT License; original processed data, metadata,
documentation, and generated figures are released under CC BY 4.0. These
licenses do not relicense third-party inputs. See `LICENSE`, `LICENSE-CODE`,
`LICENSE-DATA`, and `THIRD_PARTY_NOTICES.md` for scope and provenance. For
scholarly use, please cite the accompanying manuscript; this request is not an
additional condition of CC BY 4.0.
