import tomllib
from typing import NamedTuple


class NavigationItem(NamedTuple):
    name: str
    url: str
    icon: str
    icon_only: bool = False


def generate_navigation_for(items: list[NavigationItem], mobile: bool = False) -> str:
    # TODO: Add outlink icon
    item_template = "<a href=\"URL\" class=\"nav-link nav-icon\">NAME_OR_ICON</a>"
    html_items = []
    for item in items:
        with open(item.icon, "r") as fh:
            icon = fh.read()

        name_or_icon = icon if (mobile or item.icon_only) else item.name
        nav_html = item_template.replace("URL", item.url).replace("NAME_OR_ICON", name_or_icon)
        html_items.append(nav_html)

    items_html = "\n".join(html_items)
    classes = "mobile nav-mobile" if mobile else "desktop"
    return f"""
    <nav class="navigation-bar {classes}">
    {items_html}
    </nav>
    """


def load_navigation_definitions() -> list[NavigationItem]:
    with open("navigation.toml", "rb") as fh:
        items = tomllib.load(fh)["pages"]
    return [NavigationItem(**item) for item in items]


def generate_navigation() -> str:
    items = load_navigation_definitions()
    desktop_navigation = generate_navigation_for(items)
    mobile_navigation = generate_navigation_for(items, mobile=True)
    return desktop_navigation + mobile_navigation



