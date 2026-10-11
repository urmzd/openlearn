"""One module per `ol` verb. `olcourse.cli` imports `commands.<verb>` lazily
and calls its `main(argv) -> int`, so a verb is added by adding a file."""
