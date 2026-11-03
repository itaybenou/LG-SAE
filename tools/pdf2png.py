"""Render the paper's PDF figures to PNG for the project page.

Usage (from the repo root):  python tools/pdf2png.py [path/to/paper/figs]
"""
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\LG_SAE_NeurIPS_Camera_Ready\figs")
DST = ROOT / "static" / "figures"
DPI = 200

FIGURES = {
    "LG_SAE_overview_iclr_v4.pdf": "overview.png",
    "C_matrix_compare_v2.pdf": "c_matrix_compare.png",
    "comp_result_1.pdf": "comp_result_1.png",
    "comp_result_2.pdf": "comp_result_2.png",
    "comp_result_3.pdf": "comp_result_3.png",
    "comp_result_4.pdf": "comp_result_4.png",
    "multiscale_explanations_v2.pdf": "multiscale_explanations.png",
    "qual_seg_results.pdf": "qual_seg_results.png",
    "water_bottle_steering_fixed.pdf": "steering_example.png",
    "steerability_results_new_with_max_increasing.pdf": "steering_results.png",
    "concept_composition_violin_final_no_title.pdf": "survey_composition.png",
    "concept_naming_survey_plot_final_no_title.pdf": "survey_naming.png",
}


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in FIGURES.items():
        with fitz.open(SRC / src_name) as doc:
            pix = doc[0].get_pixmap(dpi=DPI, alpha=False)
            pix.save(DST / dst_name)
        print(f"{src_name} -> {dst_name} ({pix.width}x{pix.height})")


if __name__ == "__main__":
    main()
