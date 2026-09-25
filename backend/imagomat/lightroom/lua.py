"""Parser und Serialisierer für die Lua-Tabellen im Lightroom-Katalog.

Lightroom speichert Entwicklungseinstellungen in ``Adobe_imageDevelopSettings.text`` als
Lua-Quelltext der Form ``s = { Exposure2012 = 0.35, ToneCurvePV2012 = { 0, 0, 255, 255, }, ... }``.
"""

from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+|--\[\[.*?\]\]|--[^\n]*)
  | (?P<longstr>\[(?P<eq>=*)\[.*?\](?P=eq)\])
  | (?P<str>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
  | (?P<num>-?(?:0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<sym>[{}\[\]=,;])
    """,
    re.VERBOSE | re.DOTALL,
)

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "a": "\a", "b": "\b",
            "f": "\f", "v": "\v", "\n": "\n"}


class LuaError(ValueError):
    pass


def _unescape(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n.isdigit():
                j = i + 1
                while j < len(s) and j < i + 4 and s[j].isdigit():
                    j += 1
                out.append(chr(int(s[i + 1:j])))
                i = j
                continue
            out.append(_ESCAPES.get(n, n))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _tokens(src: str) -> list[tuple[str, str]]:
    pos, toks = 0, []
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m:
            raise LuaError(f"unerwartetes Zeichen an Position {pos}: {src[pos:pos + 20]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "eq":
            kind = "longstr"
        if kind == "ws":
            continue
        toks.append((kind, m.group(0)))
    return toks


class _Parser:
    def __init__(self, toks: list[tuple[str, str]]):
        self.toks, self.i = toks, 0

    def peek(self, k: int = 0) -> tuple[str, str]:
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else ("eof", "")

    def take(self, value: str | None = None) -> tuple[str, str]:
        t = self.peek()
        if value is not None and t[1] != value:
            raise LuaError(f"erwartet {value!r}, gefunden {t[1]!r}")
        self.i += 1
        return t

    def value(self) -> Any:
        kind, text = self.peek()
        if text == "{":
            return self.table()
        self.i += 1
        if kind == "str":
            return _unescape(text[1:-1])
        if kind == "longstr":
            m = re.fullmatch(r"\[(=*)\[(.*)\]\1\]", text, re.DOTALL)
            body = m.group(2) if m else ""
            return body[1:] if body.startswith("\n") else body
        if kind == "num":
            if text.lower().startswith(("0x", "-0x")):
                return int(text, 16)
            f = float(text)
            return int(f) if re.fullmatch(r"-?\d+", text) else f
        if kind == "name":
            if text == "true":
                return True
            if text == "false":
                return False
            if text == "nil":
                return None
            return text
        raise LuaError(f"unerwarteter Wert {text!r}")

    def table(self) -> Any:
        self.take("{")
        items: list[Any] = []
        named: dict[Any, Any] = {}
        while self.peek()[1] != "}":
            k0, t0 = self.peek()
            if k0 == "name" and self.peek(1)[1] == "=":
                self.i += 2
                named[t0] = self.value()
            elif t0 == "[":
                self.take("[")
                key = self.value()
                self.take("]")
                self.take("=")
                named[key] = self.value()
            else:
                items.append(self.value())
            if self.peek()[1] in (",", ";"):
                self.i += 1
        self.take("}")
        if named and items:
            for idx, v in enumerate(items, start=1):
                named[idx] = v
            return named
        return named if named else items


def parse_lua(src: str) -> Any:
    """Parst ``s = { ... }`` oder ``{ ... }`` und gibt dict/list zurück."""
    toks = _tokens(src.strip())
    p = _Parser(toks)
    if p.peek()[0] == "name" and p.peek(1)[1] == "=":
        p.i += 2
    return p.value()


def _lua_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _lua_val(v: Any, indent: int) -> str:
    pad = "\t" * indent
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "nil"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return repr(v)
    if isinstance(v, str):
        return _lua_str(v)
    if isinstance(v, (list, tuple)):
        if not v:
            return "{}"
        if all(not isinstance(x, (dict, list, tuple)) for x in v):
            return "{ " + ", ".join(_lua_val(x, 0) for x in v) + ", }"
        inner = "".join(f"{pad}\t{_lua_val(x, indent + 1)},\n" for x in v)
        return "{\n" + inner + pad + "}"
    if isinstance(v, dict):
        if not v:
            return "{}"
        lines = []
        for k in sorted(v, key=lambda x: (not isinstance(x, str), str(x))):
            key = k if isinstance(k, str) and re.fullmatch(r"[A-Za-z_]\w*", k) else f"[{_lua_val(k, 0)}]"
            lines.append(f"{pad}\t{key} = {_lua_val(v[k], indent + 1)},\n")
        return "{ " + "".join(lines).lstrip("\t") + pad + "}"
    raise TypeError(f"nicht serialisierbar: {type(v)}")


def dump_lua(v: Any, name: str = "s") -> str:
    return f"{name} = {_lua_val(v, 0)}\n"
