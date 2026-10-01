"""Command line entry point.

    invoice-extraction prepare                 # vendor-based splits + vendor master list
    invoice-extraction ocr --split test        # OCR a split (cached)
    invoice-extraction erp                     # start the mock ERP on :8000
    invoice-extraction run --split test -n 20  # full pipeline, writes data/runs/<ts>.jsonl
"""

import argparse
from collections import Counter
from datetime import datetime

from .config import RUNS_DIR
from .data import build_vendor_master, image_path, load_splits, load_vendor_master, make_splits
from .ocr import ocr_cached
from .schema import Route


def cmd_prepare(_: argparse.Namespace) -> None:
    splits = make_splits()
    vendors = build_vendor_master()
    print({name: len(ids) for name, ids in splits.items()}, f"vendors={len(vendors)}")


def cmd_ocr(args: argparse.Namespace) -> None:
    ids = load_splits()[args.split][: args.limit]
    for i, doc_id in enumerate(ids, 1):
        ocr_cached(doc_id, image_path(doc_id))
        print(f"\rOCR {i}/{len(ids)}", end="", flush=True)
    print()


def cmd_erp(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("invoice_extraction.erp.server:app", host="127.0.0.1", port=args.port)


def cmd_run(args: argparse.Namespace) -> None:
    from .extract import make_extractor
    from .pipeline import process_document

    ids = load_splits()[args.split][: args.limit]
    extractor = make_extractor()
    vendors = load_vendor_master()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{args.split}.jsonl"
    routes: Counter[str] = Counter()
    with out.open("w") as f:
        for doc_id in ids:
            result = process_document(doc_id, image_path(doc_id), extractor, vendors, post=not args.no_post)
            f.write(result.model_dump_json() + "\n")
            routes[result.decision.route] += 1
            flag = "AUTO " if result.decision.route is Route.AUTO_POST else "REVIEW"
            why = "; ".join(result.decision.reasons)
            print(f"{doc_id} {flag} conf={result.decision.min_confidence:.2f} {why}")

    total = sum(routes.values())
    auto = routes[Route.AUTO_POST]
    print(f"\n{auto}/{total} auto-posted, {total - auto} sent to review -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="invoice-extraction")
    sub = parser.add_subparsers(required=True)

    sub.add_parser("prepare").set_defaults(func=cmd_prepare)

    p = sub.add_parser("ocr")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("-n", "--limit", type=int)
    p.set_defaults(func=cmd_ocr)

    p = sub.add_parser("erp")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_erp)

    p = sub.add_parser("run")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("-n", "--limit", type=int)
    p.add_argument("--no-post", action="store_true", help="route only, don't call the ERP")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)
