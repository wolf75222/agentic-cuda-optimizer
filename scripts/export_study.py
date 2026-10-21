#!/usr/bin/env python3
"""Export a §16 study or reconstruct one frozen candidate without execution."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from optimizer_agent.portable_export import export_study,reconstruct_candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    export = commands.add_parser("export")
    export.add_argument("--study",type=Path,required=True)
    export.add_argument("--output",type=Path,required=True)
    reconstruct = commands.add_parser("reconstruct")
    reconstruct.add_argument("--bundle",type=Path,required=True)
    reconstruct.add_argument("--candidate",required=True,help="id or selection from export-manifest.json")
    reconstruct.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_study(args.study,args.output)
        print(json.dumps({"status":result["status"],"archive":str(args.output/"study-evidence.zip"),
            "candidates":[{"id":row["id"],"selection":row["selection"],"reconstructable":row["reconstructable"]} for row in result["candidates"]]},indent=2))
    else:
        candidate,requests,manifest = reconstruct_candidate(args.bundle,args.candidate,args.output)
        print(json.dumps({"candidate_id":candidate.candidate_id,"artifact_hash":candidate.artifact_hash,
            "requests":len(requests),"directory":str(args.output.resolve()),"executed":False,
            "export_status":manifest["status"]},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
