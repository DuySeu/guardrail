from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import httpx
from core import GuardrailConfigError, Stage, load_guardrail
from dotenv import load_dotenv
from eval.adapters import prompt_injections, toxic_chat
from eval.dataset import DatasetError, load_dataset
from eval.metrics import summarize_isolate, summarize_pipeline
from eval.report import ReportError, compare, isolate_rows, pipeline_rows, write_run
from eval.runner import run_isolate, run_pipeline

RUNS_DIR = Path("_workspace/runs")
PUBLIC_DIR = Path("datasets/public")
ADAPTERS = {
    "prompt_injections": prompt_injections.fetch,
    "toxic_chat": toxic_chat.fetch,
}


# Run one text through the guardrail and print the decision
async def check(args: argparse.Namespace) -> None:
    guardrail = load_guardrail(args.config)
    try:
        await guardrail.warmup()
        if args.stage is Stage.INPUT:
            decision = await guardrail.check_input(args.text)
        else:
            decision = await guardrail.check_output(
                args.text, user_input=args.user_input
            )
    finally:
        await guardrail.aclose()
    print(json.dumps(asdict(decision), ensure_ascii=False, indent=2))


# Benchmark a config on a dataset and write a run directory
async def evaluate(args: argparse.Namespace) -> None:
    samples = load_dataset(args.dataset)
    guardrail = load_guardrail(args.config)
    network = {
        f"{stage}/{g.name}"
        for stage in Stage
        for g in guardrail.guards(stage)
        if g.requires
    }
    try:
        if args.isolate:
            runs = await run_isolate(guardrail, samples, args.concurrency)
            summary = summarize_isolate(runs, network, args.categories)
            rows = isolate_rows(runs)
        else:
            records = await run_pipeline(guardrail, samples, args.concurrency)
            summary = summarize_pipeline(records, network)
            rows = pipeline_rows(records)
    finally:
        await guardrail.aclose()

    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    write_run(run_dir, rows, summary, args.config)
    print(run_dir)
    print((run_dir / "summary.md").read_text(encoding="utf-8"))


# Download a public dataset into datasets/public/<name>.jsonl
def fetch(args: argparse.Namespace) -> None:
    samples = ADAPTERS[args.name]()
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    out = PUBLIC_DIR / f"{args.name}.jsonl"
    with out.open("w", encoding="utf-8") as lines:
        for sample in samples:
            lines.write(sample.model_dump_json(exclude_defaults=True) + "\n")
    print(f"{out}: {len(samples)} samples")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guardrail check, benchmark and compare"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check_cmd = commands.add_parser("check", help="run one text through the guardrail")
    check_cmd.add_argument("text")
    check_cmd.add_argument(
        "--stage", type=Stage, choices=list(Stage), default=Stage.INPUT
    )
    check_cmd.add_argument("--user-input")
    check_cmd.add_argument("--config", type=Path, default=Path("config.yaml"))

    eval_cmd = commands.add_parser("eval", help="benchmark a config on a JSONL dataset")
    eval_cmd.add_argument("--dataset", type=Path, required=True)
    eval_cmd.add_argument("--config", type=Path, default=Path("config.yaml"))
    eval_cmd.add_argument("--isolate", action="store_true")
    eval_cmd.add_argument("--concurrency", type=int, default=4)
    eval_cmd.add_argument("--categories", nargs="+")

    compare_cmd = commands.add_parser("compare", help="compare two run directories")
    compare_cmd.add_argument("run_a", type=Path)
    compare_cmd.add_argument("run_b", type=Path)

    fetch_cmd = commands.add_parser("fetch", help="download a public dataset")
    fetch_cmd.add_argument("name", choices=sorted(ADAPTERS))
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            asyncio.run(check(args))
        elif args.command == "eval":
            asyncio.run(evaluate(args))
        elif args.command == "compare":
            print(compare(args.run_a, args.run_b))
        else:
            fetch(args)
    except (GuardrailConfigError, DatasetError, ReportError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
