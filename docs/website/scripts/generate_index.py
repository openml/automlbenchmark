import tomllib
from typing import NamedTuple

class Framework(NamedTuple):
    name: str
    repository: str
    icon: str

def load_header() -> str:
    with open("templates/headers.html", "r") as f:
        return f.read()

def load_footer() -> str:
    with open("templates/footer.html", "r") as f:
        return f.read()

def generate_framework_gallery():
    with open("official_frameworks.toml", "rb") as fh:
        frameworks = tomllib.load(fh)["frameworks"]
    frameworks = [Framework(**fw) for fw in frameworks]


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
    header = load_header()
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
