"""Commands: ingest once, run a daily worker, or ask a question."""

import argparse
import json
import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from pdf_rag.config import Settings
from pdf_rag.service import RagService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest")
    commands.add_parser("worker")
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    service = RagService(settings)

    def ingest():
        report = service.sync()
        print(json.dumps(report, indent=2))
        return report

    if args.command == "ingest":
        if ingest()["errors"]:
            raise SystemExit(1)
    elif args.command == "ask":
        print(json.dumps(service.ask(args.question), indent=2))
    else:
        # Startup scan catches changes made while the worker was stopped.
        ingest()
        scheduler = BlockingScheduler(timezone=settings.timezone)
        scheduler.add_job(
            ingest,
            "cron",
            hour=settings.scan_hour,
            minute=settings.scan_minute,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
        )
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass
