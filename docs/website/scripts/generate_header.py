import tomllib
from typing import NamedTuple


class NavigationItem(NamedTuple):
    name: str
    align: str
    url: str
    icon: str | None = None
    icon_only: bool = False


def generate_desktop_navigation(items: list[NavigationItem]) -> str:
    # TODO: consider `icon_only`
    item_template = "<a href=\"URL\" class=\"nav-link\">NAME</a>"
    items_html = "\n".join(
        item_template.replace("URL", item.url).replace("NAME", item.name)
        for item in items
    )
    return f"""
    <nav class="navigation-bar desktop">
    {items_html}
    </nav>
    """


def generate_mobile_navigation(items: list[NavigationItem]) -> str:
    return ""

def load_navigation_definitions() -> list[NavigationItem]:
    with open("navigation.toml", "rb") as fh:
        items = tomllib.load(fh)["pages"]
    return [NavigationItem(**item) for item in items]


def generate_navigation() -> str:
    items = load_navigation_definitions()
    desktop_navigation = generate_desktop_navigation(items)
    mobile_navigation = generate_mobile_navigation(items)
    return desktop_navigation + mobile_navigation



