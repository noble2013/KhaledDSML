"""Execute science notebook code cells (19–23) offline under release CI.

Only skip explicit first-cell "%pip install ..." (CI has already installed all
relevant extras). Each notebook runs in its own temporary working directory.
Calls to IPython display_analysis_dashboard are patched to the REAL offline
HTML renderer so both code paths and bilingual interpretation are exercised,
without relying on a Jupyter frontend. Failing code raises, never skips.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import os
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "19_final_research_studio.ipynb",
    "20_specialized_inference_and_exports.ipynb",
    "21_pp_grimmer_pcurve_studio.ipynb",
    "22_sem_shap_tdigest_bayes.ipynb",
    "23_zero_inflation_vuong_precision.ipynb",
)


def extract_cells(notebook_path):
    notebook=json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    if notebook.get("nbformat")!=4 or not isinstance(notebook.get("cells"),list):
        raise ValueError("Not a supported v4 notebook")
    code=[]
    for index,cell in enumerate(notebook["cells"]):
        if cell.get("cell_type")!="code":continue
        raw=cell.get("source")
        if not isinstance(raw,list) or any(not isinstance(x,str) for x in raw):
            raise ValueError(f"Invalid notebook code cell {index}")
        source="".join(raw)
        if not source.strip():continue
        if source.lstrip().startswith("%pip install"):
            if code or source.lstrip().splitlines()[0]!=source.strip():
                raise ValueError("Only the initial standalone %pip cell may be skipped")
            continue
        if source.lstrip().startswith(("%","!")):
            raise ValueError(f"Unexpected magics or shell command in cell {index}")
        code.append((index,source))
    if not code:
        raise ValueError("No executable research cells")
    return code


def run_notebook(notebook_path):
    import khaled_dsml as dsml
    source=Path(notebook_path)
    if not source.exists():
        raise FileNotFoundError(source)
    # Verify HTML generation, including interpretation presence, rather
    # than discarding the visualization and hiding dashboard exceptions.
    rendered=[]
    def render_headless(result,*args,**kwargs):
        html=dsml.render_analysis_dashboard(result,*args,**kwargs)
        if not isinstance(html,str) or 'dir="rtl"' not in html:
            raise AssertionError("No valid offline RTL HTML dashboard")
        rendered.append(html)
        return html

    executed=0
    state={"__name__":"__main__","__file__":str(source)}
    with tempfile.TemporaryDirectory(prefix="khaled-dsml-notebook-") as folder:
        working=Path.cwd()
        try:
            os.chdir(folder)
            with patch.object(dsml,"display_analysis_dashboard",render_headless):
                for idx,cell in extract_cells(source):
                    try:
                        exec(compile(cell,f"{source.name}:cell{idx}","exec"),state)
                    except Exception as exc:
                        raise RuntimeError(
                            f"{source.name} failed at code cell {idx}: {type(exc).__name__}: {exc}"
                        ) from exc
                    executed+=1
            if source.name.startswith("20_"):
                from openpyxl import load_workbook
                from docx import Document
                xlsx=Path("KhaledDSML_Research_Analysis.xlsx")
                docx=Path("KhaledDSML_Research_Analysis.docx")
                if not xlsx.is_file() or not docx.is_file():
                    raise AssertionError("Real bilingual Office exports were not generated")
                assert load_workbook(xlsx,read_only=True).sheetnames==["Summary","Data Table"]
                assert Document(docx).paragraphs
        finally:
            os.chdir(working)
    if not rendered:
        raise AssertionError(f"Notebook {source.name} produced no bilingual dashboards")
    return {"notebook":source.name,"code_cells_executed":executed,
            "bilingual_dashboards":len(rendered)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks",nargs="*",default=list(NOTEBOOKS),
                        help="Selected notebook filenames in the notebooks/ directory")
    args=parser.parse_args(argv)
    for name in args.notebooks:
        if name not in NOTEBOOKS:
            parser.error(f"Notebook outside reviewed smoke suite: {name}")
        output=run_notebook((ROOT/"notebooks"/name).resolve())
        print("PASS",output["notebook"],output["code_cells_executed"],
              "code cells,",output["bilingual_dashboards"],"bilingual reports")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
