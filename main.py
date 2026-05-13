"""Backward-compatible entrypoint; prefer `churn-train` after editable install."""

from telecom_churn.cli import main

if __name__ == "__main__":
    main()
