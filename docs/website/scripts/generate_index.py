def load_header() -> str:
    with open("templates/headers.html", "r") as f:
        return f.read()

def load_footer() -> str:
    with open("templates/footer.html", "r") as f:
        return f.read()

def generate_framework_gallery():
    pass

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