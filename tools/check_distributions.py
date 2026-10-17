"""Inspect built wheel AND source distribution independently.

Works with stdlib only. A successful pip wheel install does NOT guarantee
the source archive includes scientific trace data; inspect both archives.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import tarfile
import zipfile


TRACE="docs/CATALOG_256_SOURCE_TRACE_V010.tsv"


def inspect_dist(dist,version):
    path=Path(dist)
    if not path.is_dir():
        raise ValueError("Distribution directory does not exist")
    wheels=sorted(path.glob("khaled_dsml-*.whl"))
    archives=sorted(path.glob("khaled_dsml-*.tar.gz"))
    if len(wheels)!=1 or len(archives)!=1:
        raise ValueError("Require exactly one KhaledDSML wheel and one source archive")
    wheel=wheels[0];sdist=archives[0]
    if not wheel.name.startswith(f"khaled_dsml-{version}-"):
        raise ValueError("Wheel version does not match project metadata")
    if not sdist.name.startswith(f"khaled_dsml-{version}.tar.gz"):
        raise ValueError("Sdist version does not match project metadata")
    with zipfile.ZipFile(wheel) as zipped:
        names=zipped.namelist()
        pkg="khaled_dsml/__init__.py"
        if pkg not in names:
            raise ValueError("Wheel missing root import package")
        if f'__version__ = "{version}"' not in zipped.read(pkg).decode("utf-8"):
            raise ValueError("Wheel __version__ differs from metadata")
        meta=[n for n in names if n.endswith(".dist-info/METADATA")]
        if len(meta)!=1:
            raise ValueError("Wheel must contain exactly one METADATA file")
        data=zipped.read(meta[0]).decode("utf-8")
        if not re.search(r"^Name: khaled-dsml\s*$",data,re.M|re.I):
            raise ValueError("Unexpected wheel distribution name")
        if not re.search(rf"^Version: {re.escape(version)}\s*$",data,re.M):
            raise ValueError("Wheel METADATA version differs")
    with tarfile.open(sdist,"r:gz") as tar:
        names=tar.getnames()
        root=f"khaled_dsml-{version}/"
        for required in (
            "pyproject.toml","README.md",
            TRACE,"docs/ZERO_INFLATION_AND_PRECISION_0_10_AR_EN.md",
            "src/khaled_dsml/__init__.py",
        ):
            if root+required not in names:
                raise ValueError(f"Source distribution missing {required}")
        source=tar.extractfile(root+TRACE)
        if source is None:
            raise ValueError("Source trace not readable in sdist")
        content=source.read().decode("utf-8-sig").splitlines()
        if len(content)!=257:
            raise ValueError("Source distribution must contain all 256 catalogue rows")
    return {"wheel":wheel.name,"sdist":sdist.name,"source_catalogue_rows":256}


def main(argv=None):
    from importlib.util import module_from_spec, spec_from_file_location
    gate_path=Path(__file__).with_name('release_gate.py')
    spec=spec_from_file_location('khaled_dsml_release_gate',gate_path)
    if spec is None or spec.loader is None:
        raise ValueError('Release gate module missing')
    module=module_from_spec(spec)
    spec.loader.exec_module(module)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir",nargs="?",default="dist")
    args=parser.parse_args(argv)
    r=inspect_dist(args.dist_dir,module.read_versions())
    print("VALIDATED WHEEL:",r["wheel"])
    print("VALIDATED SOURCE:",r["sdist"])
    print("CATALOGUE ROWS:",r["source_catalogue_rows"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
