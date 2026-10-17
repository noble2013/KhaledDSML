"""Install wheel and sdist in TWO clean virtualenvs and run user-facing API.

Unlike importing the checkout from repository root, these checks run Python
in isolated mode (-I), outside the source tree. Both distribution forms
must import, have consistent metadata and compute representative statistics.
No external publication or TestPyPI credentials are required.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import venv


ROOT=Path(__file__).resolve().parents[1]
VERIFY=r"""
import importlib.metadata as metadata
from pathlib import Path
import sys
import numpy as np
import khaled_dsml as dsml

expected=sys.argv[1]
assert dsml.__version__ == expected
assert metadata.version("khaled-dsml") == expected
source=Path(dsml.__file__).resolve()
prefix=Path(sys.prefix).resolve()
assert source.is_relative_to(prefix), (source,prefix)
assert source.name == "__init__.py"

rng=np.random.default_rng(2026)
counts=rng.poisson(1.5,size=600)
poisson=dsml.poisson_excess_zero_score(counts)
assert 0 <= poisson["p_value"] <= 1 and poisson["n_samples"]==len(counts)
bayes=dsml.beta_binomial_posterior(18,50)
assert 0 < bayes["posterior_mean"] < 1
precision=dsml.post_study_precision_scenarios(.25,.1,[0.,.2,.4])
assert precision["observed_power_computed"] is False
assert len(precision["table"]) == 3
print("VALIDATED",expected,str(source),"Poisson, Bayes, precision")
"""


def _python_bin(root):
    path=Path(root)
    return path/("Scripts/python.exe" if os.name=="nt" else "bin/python")


def _run(args,*,cwd,env):
    subprocess.run([str(x) for x in args],check=True,cwd=str(cwd),env=env)


def find_archives(dist_dir,version):
    path=Path(dist_dir).resolve()
    wheels=list(path.glob(f"khaled_dsml-{version}-*.whl"))
    sdist=path/f"khaled_dsml-{version}.tar.gz"
    if len(wheels)!=1 or not sdist.is_file():
        raise ValueError("Expected one matching wheel and sdist for current version")
    return {"wheel":wheels[0],"sdist":sdist}


def run_install_checks(dist_dir,version,*,modes=("wheel","sdist")):
    files=find_archives(dist_dir,version)
    if not modes or any(mode not in files for mode in modes):
        raise ValueError("Requested mode must be wheel or sdist")
    results=[]
    with tempfile.TemporaryDirectory(prefix="khaled-distribution-smoke-") as temp:
        isolated=Path(temp)
        env=os.environ.copy()
        env.pop("PYTHONPATH",None)
        env["PYTHONNOUSERSITE"]="1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"]="1"
        for mode in modes:
            env_dir=isolated/f"env-{mode}"
            venv.create(env_dir,with_pip=True,clear=True)
            python=_python_bin(env_dir)
            _run([python,"-m","pip","install","--no-input",str(files[mode])],
                 cwd=isolated,env=env)
            _run([python,"-m","pip","check"],cwd=isolated,env=env)
            _run([python,"-I","-c",VERIFY,version],cwd=isolated,env=env)
            # Installed console entry point must operate outside repository too.
            binary=env_dir/("Scripts/khaled-dsml.exe" if os.name=="nt" else "bin/khaled-dsml")
            _run([binary,"--help"],cwd=isolated,env=env)
            results.append({"mode":mode,"archive":files[mode].name,
                            "version":version,"installed_outside_source":True})
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir",nargs="?",default="dist")
    parser.add_argument("--mode",choices=("wheel","sdist"),action="append",
                        dest="modes",help="Repeat to select both package formats")
    args=parser.parse_args(argv)
    from importlib.util import module_from_spec, spec_from_file_location
    spec=spec_from_file_location("khaled_release_gate",ROOT/"tools/release_gate.py")
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import the trusted source version gate")
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    version=mod.read_versions()
    for item in run_install_checks(args.dist_dir,version,
                                   modes=args.modes or ("wheel","sdist")):
        print("PASS",item["mode"],item["archive"],
              item["version"],"in clean isolated environment")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
