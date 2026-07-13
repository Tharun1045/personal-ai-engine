import re
import hashlib
import unicodedata


def clean_content(content: str) -> tuple[str, str]:
    """Clean markdown/html content and return (cleaned_text, content_hash)."""
    if not content:
        return "", ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", content)

    # Remove script and style blocks (if any leaked into markdown)
    text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?>.*?</style>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove boilerplate patterns (nav, footer, cookie banners)
    boilerplate_patterns = [
        r"(?i)(accept cookies|cookie policy|privacy policy)",
        r"(?i)(skip to content|skip to main)",
        r"(?i)(subscribe to newsletter|sign up for our newsletter)",
        r"(?i)(all rights reserved|© \d{4})",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)

    # Whitespace normalization (preserve newlines but clean spaces/tabs)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
        elif cleaned_lines and cleaned_lines[-1] != "":
            # preserve paragraph breaks (double newline)
            cleaned_lines.append("")

    text = "\n".join(cleaned_lines)

    # Remove repeated blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Stable content hashing
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return text, content_hash
