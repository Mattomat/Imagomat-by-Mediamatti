"""rawpy so bauen, dass LibRaw keine Zusatzbibliotheken (Little CMS, libjpeg, Jasper) braucht.

Sonst verlinkt der Build auf dem CI-Rechner vorhandene Homebrew-Bibliotheken, die auf den Macs der
Nutzer fehlen ("Library not loaded: /opt/homebrew/opt/little-cms2/..."). Imagomat braucht diese Teile
nicht (keine ICC-Ausgabe, kein verlustbehaftetes DNG, kein JPEG 2000).

Aufruf: python rawpy_no_external_deps.py <rawpy-Quellordner>/setup.py
"""

import sys
from pathlib import Path

EXTRA = ('"-DENABLE_LCMS=OFF", "-DENABLE_JASPER=OFF", "-DCMAKE_DISABLE_FIND_PACKAGE_JPEG=TRUE", '
         '"-DCMAKE_DISABLE_FIND_PACKAGE_LCMS2=TRUE", "-DCMAKE_DISABLE_FIND_PACKAGE_LCMS=TRUE", '
         '"-DCMAKE_DISABLE_FIND_PACKAGE_Jasper=TRUE",')

p = Path(sys.argv[1])
lines = p.read_text().splitlines(keepends=True)
out, n = [], 0
for line in lines:
    out.append(line)
    if line.strip() == '"-DENABLE_RAWSPEED=OFF",':
        indent = line[: len(line) - len(line.lstrip())]
        out.append(f"{indent}{EXTRA}\n")
        n += 1
if n != 1:
    sys.exit(f"setup.py: Stelle für die CMake-Optionen nicht eindeutig gefunden ({n}x)")
p.write_text("".join(out))
print("CMake: LibRaw ohne externe Bibliotheken")
