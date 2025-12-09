#!/usr/bin/env python3
"""
Remove Chinese characters from comments and triple-quoted strings in Python files.
Creates a `.bak` backup for each modified file.

Usage:
  python3 scripts/remove_chinese_comments.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
TRIPLE_RE = re.compile(r'("""|\'\'\')(.*?)(\1)', re.DOTALL)


def process_text(text: str) -> str:
    orig = text
    # 1) Remove line comments that contain any Chinese characters (remove the comment part)
    text = re.sub(r"#.*[\u4e00-\u9fff].*$", "", text, flags=re.MULTILINE)

    # 2) For inline comments that mix Chinese and ascii, remove only Chinese chars from comment
    
    def _clean_inline(m):
        comment = m.group(0)
        cleaned = re.sub(CHINESE_RE, "", comment)
        # Collapse multiple spaces after removing
        return cleaned

    text = re.sub(r"#.*", _clean_inline, text)

    # 3) Handle triple-quoted strings (docstrings / multiline strings)
    def _clean_triple(m):
        quote = m.group(1)
        body = m.group(2)
        if CHINESE_RE.search(body):
            new_body = re.sub(CHINESE_RE, "", body)
            if new_body.strip() == "":
                # remove the entire triple-quoted block
                return ""
            return quote + new_body + quote
        return m.group(0)

    text = TRIPLE_RE.sub(_clean_triple, text)

    # If nothing changed, return original to avoid needless writes
    return text if text != orig else orig


def main():
    py_files = list(ROOT.rglob('*.py'))
    changed = []
    for p in py_files:
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        new_text = process_text(text)
        if new_text != text:
            bak = p.with_suffix(p.suffix + '.bak')
            bak.write_text(text, encoding='utf-8')
            p.write_text(new_text, encoding='utf-8')
            changed.append(str(p.relative_to(ROOT)))
    if changed:
        print('Modified files:')
        for f in changed:
            print(' -', f)
    else:
        print('No changes made.')


if __name__ == '__main__':
    main()
