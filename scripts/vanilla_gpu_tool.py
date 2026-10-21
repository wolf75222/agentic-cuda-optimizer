#!/usr/bin/env python3
"""Read properties or submit workspace CUDA/launch paths to parent GPU tools.

launch.json is a LaunchRule, e.g. {"kind":"square_2d","block":[16,16,1]},
or {"launch_rule":{...},"parameters":{...}}. No evaluator configuration, oracle,
input buffers, final cases, or benchmark protocol can be supplied by the child.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("properties", "validate", "benchmark"))
    parser.add_argument("--endpoint", default=os.environ.get("SERVICE_ENDPOINT"))
    parser.add_argument("--token", default=os.environ.get("SERVICE_TOKEN"))
    parser.add_argument("--source", help="CUDA source path inside the isolated child workspace")
    parser.add_argument("--launch", help="LaunchRule JSON path inside the isolated child workspace")
    args = parser.parse_args(argv)
    endpoint = urllib.parse.urlsplit(args.endpoint or "")
    if endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1" or not endpoint.port or not args.token:
        parser.error("A loopback HTTP endpoint and bearer token are required")
    if args.operation != "properties" and not (args.source and args.launch):
        parser.error("validate/benchmark require --source and --launch")
    data = None if args.operation == "properties" else json.dumps({"source": args.source, "launch": args.launch}).encode()
    request = urllib.request.Request(args.endpoint.rstrip("/") + "/" + args.operation,
        data=data, headers={"Authorization": "Bearer " + args.token, "Content-Type": "application/json"},
        method="GET" if data is None else "POST")
    try:
        # Parent worker operations have their own technical CUDA/SSH timeouts.
        # This client adds no model, campaign, call-count, or resource budget.
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        try:
            result = json.load(error)
        except ValueError:
            result = {"ok": False, "error": str(error)}
    except (OSError, ValueError) as error:
        result = {"ok": False, "error": str(error)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
