def load_header() -> str:
    with open("templates/headers.html", "r") as f:
        return f.read()

def load_footer() -> str:
    with open("templates/footer.html", "r") as f:
        return f.read()

def generate_framework_gallery():
    template = """
    <a href=REPOSITORY target="_blank" class="framework-logo">
    <img src=ICON title=NAME/>
    </a>
    """
    frameworks = [
        template.replace(
            "REPOSITORY", repository
        ).replace(
            "ICON", icon
        ).replace("NAME", name)
        for name, (repository, icon) in frameworks.items()
    ]
    return "\n".join(frameworks)


def generate_main_page() -> str:
    header = load_header()
    footer = load_footer()

    with open("templates/index_template.html", "r") as f:
        main_content = f.read()
    # framework_gallery = generate_framework_gallery()
    # body_html = main_content.substitute(
    #     "framework_gallery", framework_gallery
    # )
    main_content = main_content.replace(
        "<!--NAV-->", header
    ).replace("<!--FOOTER-->", footer)

    return main_content


if __name__ == "__main__":
    main_html = generate_main_page()
    with open("index_new.html", "w") as f:
        f.write(main_html)