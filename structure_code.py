import os

IGNORE_DIRS = {
    "__pycache__", ".git", ".idea", ".vscode", "venv", ".venv",
    "env", "build", "dist", "node_modules", "site-packages", "lib"
}
IGNORE_FILE_EXTENSIONS = {".pyc", ".pyo", ".pyd", ".so", ".dll"}

def build_tree_html(startpath, level=0):
    """Recursively build the directory tree with collapsible HTML."""
    lines = []
    try:
        items = sorted(os.listdir(startpath))
    except PermissionError:
        return lines

    for item in items:
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            if item in IGNORE_DIRS:
                continue
            indent = "  " * level
            lines.append(f"{indent}<details><summary>📁 {item}/</summary>")
            lines.extend(build_tree_html(path, level + 1))
            lines.append(f"{indent}</details>")
        else:
            ext = os.path.splitext(item)[1]
            if ext in IGNORE_FILE_EXTENSIONS:
                continue
            indent = "  " * (level + 1)
            lines.append(f"{indent}📄 {item}<br>")
    return lines


def save_tree_to_markdown(startpath, output_file="project_directory.md"):
    """Generate a Markdown file with collapsible folders."""
    header = f"# 📦 Project Directory Structure\n\n**Root:** `{os.path.abspath(startpath)}`\n\n"
    tree_html = "\n".join(build_tree_html(startpath))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(header + tree_html)
    print(f"✅ Collapsible project structure saved to '{output_file}'")


if __name__ == "__main__":
    save_tree_to_markdown(".")
