"""Bundle colour-tool.src.html with scene.js and styles.js into the self-contained colour-tool.html."""
from pathlib import Path
here = Path(__file__).parent
page = (here / "colour-tool.src.html").read_text()
page = page.replace("/*SCENE*/", (here / "scene.js").read_text()).replace("/*STYLES*/", (here / "styles.js").read_text())
(here / "colour-tool.html").write_text(page)
