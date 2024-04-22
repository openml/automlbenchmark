import tomllib
from typing import NamedTuple, Sequence

from generate_header import generate_navigation


class Paper(NamedTuple):
    title: str
    abstract: str
    pdf: str
    arxiv: str
    venue: str
    year: int


class Framework(NamedTuple):
    name: str
    repository: str
    icon: str
    summary: str
    papers: Sequence[Paper]

def load_navigation() -> str:
    return generate_navigation()

def load_footer() -> str:
    with open("templates/footer.html", "r") as f:
        return f.read()

def generate_framework_gallery():
    with open("official_frameworks.toml", "rb") as fh:
        frameworks = tomllib.load(fh)["frameworks"]

    frameworks = [
        Framework(
            **{attr: val for attr, val in fw.items() if attr != "papers"},
            papers=tuple(Paper(**paper) for paper in fw.get("papers", [])),
        )
        for fw in frameworks
    ]

    template = """
    <a href=\"REPOSITORY\" target="_blank" class="framework-logo">
    <img src=\"ICON\" title=\"NAME\"/>
    </a>
    """
    frameworks = [
        template.replace(
            "REPOSITORY", fw.repository
        ).replace(
            "ICON", fw.icon
        ).replace("NAME", fw.name)
        for fw in frameworks
    ]
    return "\n".join(frameworks)


def generate_main_page() -> str:
    header = load_navigation()
    footer = load_footer()

    with open("templates/index_template.html", "r") as f:
        main_content = f.read()

    framework_gallery = generate_framework_gallery()
    main_content = main_content.replace(
        "<!--NAV-->", header
    ).replace(
        "<!--FOOTER-->", footer
    ).replace("<!--FRAMEWORK_GALLERY-->", framework_gallery)

    return main_content


if __name__ == "__main__":
    main_html = generate_main_page()
    with open("index_new.html", "w") as f:
        f.write(main_html)
