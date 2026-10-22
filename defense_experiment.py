"""Run the original source-only instruction defense with the same experiment CLI."""
from run_tasks import main


if __name__ == "__main__":
    raise SystemExit(main(defense=True))
