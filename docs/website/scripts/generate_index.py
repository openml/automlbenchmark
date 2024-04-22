from pathlib import Path
from string import Template

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
    template = Template("""
    <a href=\"${repository}\" target="_blank" class="framework-logo">
    <img src=\"${icon}\" title=\"${name}\"/>
    </a>
    """)
    framework_icon_html = [
        template.substitute(fw._asdict())
        for fw in frameworks
    ]
    return "\n".join(framework_icon_html)


def generate_main_page(frameworks: Sequence[Framework]) -> str:
    header = load_navigation()
    footer = load_footer()

    with open("templates/index_template.html", "r") as f:
        main_content = Template(f.read())

    framework_gallery = generate_framework_gallery(frameworks)
    return main_content.substitute(
        **dict(
            navigation=header,
            footer=footer,
            framework_gallery=framework_gallery,
        )
    )


def generate_framework_list(
        frameworks: Sequence[Framework],
        framework_card_template: Template,
        framework_paper_template: Template,
) -> str:
    framework_cards = []
    for framework in frameworks:
        paper_list = "\n".join(
            framework_paper_template.substitute(paper._asdict())
            for paper in framework.papers
        )
        framework_card = framework_card_template.substitute(
            framework._asdict() | {"paper_list": paper_list},
        )
        framework_cards.append(framework_card)

    return "\n".join(framework_cards)


def generate_framework_page(frameworks: Sequence[Framework]) -> str:
    navigation = load_navigation()
    footer = load_footer()

    with open("templates/frameworks_template.html", "r") as f:
        main_content = Template(f.read())

    with open("templates/framework_card_template.html", "r") as f:
        framework_card_template = Template(f.read())

    with open("templates/framework_card_paper_template.html") as f:
        framework_paper_template = Template(f.read())

    framework_cards = generate_framework_list(
        frameworks,
        framework_card_template,
        framework_paper_template,

    )

    main_content = main_content.substitute(**dict(
            navigation=navigation,
            footer=footer,
            framework_cards=framework_cards,
        )
    )

    return main_content


if __name__ == "__main__":
    frameworks = load_framework_definitions()
    main_html = generate_main_page(frameworks)
    with open("index_new.html", "w") as f:
        f.write(main_html)

    framework_html = generate_framework_page(frameworks)
    with open("frameworks.html", "w") as f:
        f.write(framework_html)
