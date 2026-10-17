"""Deterministic educational sketches for approximate distinctness and similarity.

These are *bounded-memory approximations*. They do not certify equality,
uniqueness or a complete search for near-duplicates. Cryptographic hashing is
used only for reproducible item dispersal, not security/authentication.
"""
from __future__ import annotations

import hashlib
import math
from itertools import combinations

import numpy as np
import pandas as pd


def _item_bytes(item):
    if isinstance(item,str):return b"s:"+item.encode("utf-8")
    if isinstance(item,int) and not isinstance(item,bool):return b"i:"+str(item).encode()
    if isinstance(item,bytes):return b"b:"+item
    raise ValueError("Sketch inputs must be text, bytes or integer tokens")


def hyperloglog_cardinality(items, *, precision=10):
    """Classical HLL estimator using 160-bit stable SHA1 hashes.

    m=2^precision registers. Includes linear-counting small-cardinality
    correction; no high-cardinality correction is needed for SHA1's 160-bit
    range. Relative error ≈1.04/sqrt(m) under uniform independent hashing.
    Actual errors vary, particularly under nonrandom/adversarial inputs.
    """
    if type(precision) is not int or not 4<=precision<=16:
        raise ValueError("precision must be integer 4..16")
    m=1<<precision
    registers=np.zeros(m,dtype=np.uint8)
    total=0
    for item in items:
        digest=int.from_bytes(hashlib.sha1(_item_bytes(item)).digest(),"big")
        bucket=digest>>(160-precision)
        remaining=digest&((1<<(160-precision))-1)
        rho=(160-precision)-remaining.bit_length()+1
        if rho>registers[bucket]:registers[bucket]=rho
        total+=1
    z=float(np.exp2(-registers.astype(np.int16)).sum())
    alpha=.673 if m==16 else .697 if m==32 else .709 if m==64 else .7213/(1+1.079/m)
    estimate=float(alpha*m*m/z)
    vacant=int(np.count_nonzero(registers==0))
    if estimate<=2.5*m and vacant:
        estimate=float(m*math.log(m/vacant))
    if total==0:estimate=0.
    return {
        "test":"HyperLogLog approximate cardinality",
        "n_items_seen":total,"estimate_distinct":estimate,"register_count":m,
        "precision":precision,"relative_standard_error_nominal":float(1.04/np.sqrt(m)),
        "interpretation_ar":(
            "تقدير تقريبي لعدد العناصر المميزة باستخدام HyperLogLog "
            "وبصمة ثابتة. الخطأ النظري يفترض توزيع تجزئة قريبًا من "
            "الانتظام، ولا يضمن الدقة لكل عينة. لا تُستعاد العناصر الأصلية."
        ),
        "interpretation_en":(
            "Approximate distinct-item cardinality with deterministic "
            "HyperLogLog registers. Nominal relative standard error assumes "
            "approximately uniform hashes and is not a confidence guarantee "
            "on arbitrary inputs."
        ),
    }


def minhash_lsh_candidates(documents, *, num_perm=64, bands=8):
    """MinHash signatures and banded LSH near-duplicate candidate pairs.

    Documents are sequences/sets of tokens; identical-token sets have
    identical signatures and will always be LSH candidates. Non-identical
    sets may yield misses or false positives depending on bands/rows;
    no exact Jaccard search guarantee. Hash-based MinHash here is an
    approximation with SHA1 salted independent permutations.
    """
    if (not isinstance(documents,(list,tuple)) or not 2<=len(documents)<=1000
        or type(num_perm) is not int or not 16<=num_perm<=256
        or type(bands) is not int or not 1<=bands<=num_perm
        or num_perm%bands):
        raise ValueError("Require 2..1000 documents, 16..256 hash functions and divisor bands")
    sets=[]
    for doc in documents:
        if isinstance(doc,(str,bytes)):
            raise ValueError("Each document must be an iterable of tokens, not raw text")
        try: tokens={_item_bytes(v) for v in doc}
        except (TypeError,ValueError) as exc:
            raise ValueError("Every document must contain supported tokens") from exc
        if not tokens:raise ValueError("MinHash requires nonempty token sets")
        sets.append(tokens)
    signatures=[]
    for tokens in sets:
        sig=[]
        for j in range(num_perm):
            salt=j.to_bytes(4,"big")
            sig.append(min(int.from_bytes(hashlib.sha1(salt+b"|"+tok).digest(),"big")
                           for tok in tokens))
        signatures.append(sig)
    rows=num_perm//bands
    buckets={}
    for i,sig in enumerate(signatures):
        for b in range(bands):
            band=tuple(sig[b*rows:(b+1)*rows])
            buckets.setdefault((b,band),[]).append(i)
    pairs=set()
    for matches in buckets.values():
        if len(matches)>1:
            pairs.update(combinations(matches,2))
    table=pd.DataFrame([
        {"left_document":i,"right_document":j,
         "minhash_jaccard_estimate":sum(a==b for a,b in zip(signatures[i],signatures[j]))/num_perm}
        for i,j in sorted(pairs)
    ],columns=["left_document","right_document","minhash_jaccard_estimate"])
    return {
        "test":"Banded MinHash LSH candidate discovery",
        "n_documents":len(sets),"n_permutations":num_perm,"n_bands":bands,
        "n_candidate_pairs":len(pairs),"table":table,
        "interpretation_ar":(
            "ترشيح أزواج نصوص متشابهة باستخدام MinHash وLSH؛ "
            "الأزواج المرشحة ليست مطابقة مؤكدة، وقد تُفوت أزواج "
            "متشابهة بحسب عدد الحزم ودوال التجزئة. المقارنة مع "
            "Jaccard الدقيقة لازمة لقرار المطابقة النهائي."
        ),
        "interpretation_en":(
            "Banding MinHash estimates Jaccard similarity and returns "
            "potential near-duplicate pairs. Hash collisions, false "
            "positives and false negatives are possible; verify candidates "
            "with exact Jaccard if needed. Not guaranteed exhaustive."
        ),
    }
