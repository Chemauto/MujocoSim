from pathlib import Path

p = Path("/tmp/rmext/extension/out/extension.js")
src = p.read_text(encoding="utf-8")
old = "return `$(ellipsis) ${(memoryUsedWithUnits).toFixed(2)}/${(memoryTotalWithUnits).toFixed(2)} ${unit}`;"
new = (
    "return `$(ellipsis) ${(memoryUsedWithUnits).toFixed(2)}/"
    "${(memoryTotalWithUnits).toFixed(2)}${unit} "
    "(${(100 * memoryData.active / memoryData.total).toFixed(1)}%)`;"
)
assert old in src, "extension.js format string not found"
p.write_text(src.replace(old, new), encoding="utf-8")
print("resmon patched: MEM 显示 used/totalGB (xx.x%)")
