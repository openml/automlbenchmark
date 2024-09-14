import tomllib
from pathlib import Path
from typing import NamedTuple, Iterable


class NavigationItem(NamedTuple):
    name: str
    url: str
    icon: str
    icon_only: bool = False


def generate_navigation_for(items: Iterable[NavigationItem], mobile: bool = False) -> str:
    # TODO: Add outlink icon
    item_template = "<a href=\"URL\" class=\"nav-link nav-icon\">NAME_OR_ICON</a>"
    html_items = []
    for item in items:
        nav_html = generate_nav_item_html(item, item_template, mobile)
        html_items.append(nav_html)

    items_html = "\n".join(html_items)
    classes = "mobile nav-mobile" if mobile else "desktop"
    return f"""
    <nav class="navigation-bar {classes}">
    {items_html}
    </nav>
    """


def generate_nav_item_html(item, item_template, mobile):
    requires_icon = not mobile and not item.icon_only
    if not requires_icon:
        return item_template.replace("URL", item.url).replace("NAME_OR_ICON", item.name)

    with open(item.icon, "r") as fh:
        icon = fh.read()
    return item_template.replace("URL", item.url).replace("NAME_OR_ICON", icon)


def load_navigation_definitions(configuration_file: Path = Path("navigation.toml")) -> list[NavigationItem]:
    with configuration_file.open("rb") as fh:
        items = tomllib.load(fh)["pages"]
    return [NavigationItem(**item) for item in items]


def generate_navigation() -> str:
    items = load_navigation_definitions()
    desktop_navigation = generate_navigation_for(items)
    mobile_navigation = generate_navigation_for(items, mobile=True)
    return desktop_navigation + mobile_navigation



