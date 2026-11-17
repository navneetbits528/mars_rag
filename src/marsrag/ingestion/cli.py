import argparse,json
from marsrag.db import migrate
from marsrag.ingestion.pipeline import estimate_ingest,ingest_entries,load_manifest

def main():
    parser=argparse.ArgumentParser(description="Ingest NASA Mars mission PDFs.")
    parser.add_argument("--manifest",default="manifest.json")
    parser.add_argument("--report-no",action="append")
    parser.add_argument("--estimate-only",action="store_true")
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--migrate",action="store_true")
    args=parser.parse_args()
    entries=load_manifest(args.manifest)
    if args.report_no:
        wanted=set(args.report_no)
        entries=[e for e in entries if e["report_no"] in wanted]
        missing=wanted-{e["report_no"] for e in entries}
        if missing: raise SystemExit(f"Unknown report number(s): {', '.join(sorted(missing))}")
    if args.migrate: migrate()
    result=estimate_ingest(entries) if args.estimate_only else ingest_entries(entries,args.force)
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
