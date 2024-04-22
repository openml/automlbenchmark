from pathlib import Path

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
    authors: str


class Framework(NamedTuple):
    name: str
    repository: str
    icon: str
    summary: str
    papers: Sequence[Paper]


def load_framework_definitions(definition_file: Path = Path("official_frameworks.toml")) -> list[Framework]:
    with definition_file.open("rb") as fh:
        frameworks = tomllib.load(fh)["frameworks"]

    return [
        Framework(
            **{attr: val for attr, val in fw.items() if attr != "papers"},
            papers=tuple(Paper(**paper) for paper in fw.get("papers", [])),
        )
        for fw in frameworks
    ]


def load_navigation() -> str:
    return generate_navigation()


def load_footer() -> str:
    with open("templates/footer.html", "r") as f:
        return f.read()


def generate_framework_gallery(frameworks: Sequence[Framework]) -> str:
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


def generate_main_page(frameworks: Sequence[Framework]) -> str:
    header = load_navigation()
    footer = load_footer()

    with open("templates/index_template.html", "r") as f:
        main_content = f.read()

    framework_gallery = generate_framework_gallery(frameworks)
    main_content = main_content.replace(
        "<!--NAV-->", header
    ).replace(
        "<!--FOOTER-->", footer
    ).replace("<!--FRAMEWORK_GALLERY-->", framework_gallery)

    return main_content


def generate_framework_list(
        frameworks: Sequence[Framework],
        framework_card_template: str,
        framework_paper_template: str,
) -> str:
    framework_cards = []
    for framework in frameworks:
        framework_card = framework_card_template.replace(
            "NAME", framework.name,
        ).replace(
            "ICON", framework.icon,
        ).replace(
            "REPOSITORY", framework.repository,
        ).replace(
            "DOCUMENTATION", framework.repository,  # TODO
        )
        framework_card.replace(
            "<!--PAPERS-->", "\n".join(
                framework_paper_template.replace(
                    "TITLE", paper.title,
                )
                for paper in framework.papers
        )
        )


    return "\n".join(
        framework_card_template
        for framework in frameworks
    )

def generate_framework_page() -> str:
    header = load_navigation()
    footer = load_footer()

    with open("templates/framework_template.html", "r") as f:
        main_content = f.read()

    framework_cards = generate_framework_list()

    main_content = main_content.replace(
        "<!--NAV-->", header
    ).replace(
        "<!--FOOTER-->", footer
    ).replace("<!--FRAMEWORK_CARDS-->", framework_cards)

    return main_content


if __name__ == "__main__":
    frameworks = load_framework_definitions()
    main_html = generate_main_page(frameworks)
    with open("index_new.html", "w") as f:
        f.write(main_html)
