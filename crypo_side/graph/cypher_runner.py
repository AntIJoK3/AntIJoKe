from pathlib import Path


def split_cypher_blocks(text: str) -> list[str]:
    """按分号拆分语句，忽略空行与纯注释行。"""
    parts: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        buf.append(line)
        if stripped.endswith(";"):
            block = "\n".join(buf).strip()
            if block:
                parts.append(block.rstrip(";").strip() + ";")
            buf = []
    tail = "\n".join(buf).strip()
    if tail:
        parts.append(tail if tail.endswith(";") else tail + ";")
    return parts


def load_cypher_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_cypher_file(session, path: Path) -> None:
    text = load_cypher_file(path)
    for stmt in split_cypher_blocks(text):
        if not stmt.strip() or stmt.strip() == ";":
            continue
        session.run(stmt)


def resources_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "resources" / "cypher"
