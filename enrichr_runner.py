"""
Standalone enrichr runner — executed in venv310 via subprocess.
Reads ligand lists from stdin (JSON), writes results to stdout (JSON).
Called as: python3.10 enrichr_runner.py < input.json > output.json
"""

import sys, json, time
import gseapy as gp

def run_enrichr(gene_list, atom_idx, gene_sets_priority=("MSigDB_Hallmark_2020", "KEGG_2021_Human")):
    if not gene_list:
        return {"atom": atom_idx, "pathways": [], "error": "empty gene list"}

    for gene_sets in gene_sets_priority:
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_sets,
                organism="Human",
                outdir=None,
                verbose=False,
            )
            df = enr.results
            if df is None or len(df) == 0:
                continue
            # Sort by adjusted p-value
            df = df.sort_values("Adjusted P-value")
            top3 = df.head(3)[["Term", "Adjusted P-value", "Overlap", "Genes"]].to_dict(orient="records")
            return {"atom": atom_idx, "gene_sets": gene_sets, "pathways": top3, "error": None}
        except Exception as e:
            time.sleep(2)   # rate-limit pause
            last_err = str(e)

    return {"atom": atom_idx, "pathways": [], "error": last_err}


if __name__ == "__main__":
    payload = json.load(sys.stdin)   # [{atom: int, genes: [str]}, ...]
    results = []
    for item in payload:
        r = run_enrichr(item["genes"], item["atom"])
        results.append(r)
        sys.stderr.write(f"  Atom {item['atom']} done\n")
        sys.stderr.flush()
        time.sleep(1)   # be gentle with Enrichr API
    json.dump(results, sys.stdout)
