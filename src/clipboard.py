import pyperclip


def get_clipboard_text() -> str:
    clipboard_content = pyperclip.paste()

    if isinstance(clipboard_content, str):
        return clipboard_content

    return ""
