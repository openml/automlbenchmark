import tomllib
from typing import NamedTuple


class NavigationItem(NamedTuple):
    name: str
    url: str
    icon: str | None = None
    icon_only: bool = False


def generate_desktop_navigation(items: list[NavigationItem]) -> str:
    # TODO: Add outlink icon
    item_template = "<a href=\"URL\" class=\"nav-link nav-icon\">NAME_OR_ICON</a>"
    html_items = []
    for item in items:
        icon = ""
        if item.icon and item.icon.endswith(".svg"):
            with open(item.icon, "r") as fh:
                icon = fh.read()

        name_or_icon = icon if item.icon_only else item.name
        nav_html = item_template.replace("URL", item.url).replace("NAME_OR_ICON", name_or_icon)
        html_items.append(nav_html)

    items_html = "\n".join(html_items)
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



