"""Export the cleaned text used by the uDream index before chunking."""

from pathlib import Path

from rag import records


def main():
    project = Path(__file__).resolve().parent
    output = project / "processed"
    output.mkdir(exist_ok=True)

    sources = sorted((project / "documents").glob("*"))
    for source in sources:
        if source.suffix not in {".pdf", ".txt"}:
            continue

        sections = [
            f"=== {section} ===\n\n{body}"
            for _, _, section, body in records(source)
        ]
        target = output / f"{source.name}.txt"
        target.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
        print(f"Wrote {target.relative_to(project)}")


if __name__ == "__main__":
    main()
