import os
import json
import time

from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

BASE_DIR = Path("data/base_questions")

# Using English to translate to all other languages.
SOURCE_LANG: str = "en_us"

# This will skip English for translating.
SKIP_LANGS: set = {"en_us"}

MODEL_NAME: str = "gpt-5.1"

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_openai(prompt: str) -> str:
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict translation engine. "
                        "You never add explanations. "
                        "You only return translated text."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.output_text.strip()

    except Exception as e:
        print(f"Erro OpenAI: {e}")
        time.sleep(5)
        return call_openai(prompt)


def translate_filename(
    filename: str,
    target_lang_name: str
) -> str:
    name: str = filename.replace(".md", "")

    prompt: str = f"""
        You are a translation engine.

        TARGET LANGUAGE: {target_lang_name}

        STRICT RULES:
        - Output ONLY the translated filename
        - Translate completely
        - Use ONLY the target language
        - Keep lowercase
        - Replace spaces with underscores
        - Only use a-z, 0-9, _, -
        - No explanations
        - No file extension

        Text:
        {name}
    """

    translated = call_openai(prompt)

    translated = translated.lower().replace(" ", "_")
    translated = "".join(
        c for c in translated
        if c.isalnum() or c in "_-"
    )

    return translated + ".md"


def translate_questions(
    content: str,
    target_lang_name: str
) -> str:

    prompt: str = f"""
        You are an EXTREMELY STRICT deterministic translation engine.

        TARGET LANGUAGE: {target_lang_name}

        CRITICAL HARD RULES:
        - Translate EVERYTHING to {target_lang_name}
        - EVERY sentence MUST be fully translated
        - ZERO Portuguese words are allowed unless target language is Portuguese
        - ZERO English words are allowed unless target language is English
        - NEVER keep original text
        - NEVER preserve source-language words
        - NEVER partially translate
        - NEVER mix languages
        - NEVER explain
        - NEVER add notes
        - NEVER summarize
        - NEVER skip lines
        - NEVER output markdown fences
        - NEVER add headers

        STRUCTURE RULES:
        - Preserve numbering EXACTLY
        - Preserve line breaks EXACTLY
        - Preserve punctuation EXACTLY
        - Preserve markdown formatting EXACTLY
        - Keep one question per line
        - Keep identical ordering
        - Preserve quotation marks
        - Preserve list numbering

        VALIDATION RULE:
        Before returning the answer, internally verify:
        1. Every sentence is written entirely in {target_lang_name}
        2. No Portuguese remains
        3. No English remains
        4. All 20 questions exist
        5. No missing lines

        If validation fails, regenerate internally until fully compliant.

        CONTENT TO TRANSLATE:
        {content}
    """

    return call_openai(prompt)


def get_single_md_file(
    folder: Path
) -> Path:
    md_files = list(folder.glob("*.md"))

    if len(md_files) == 0:
        raise FileNotFoundError(f"No .md file found on {folder}")

    if len(md_files) > 1:
        raise ValueError(f"More than one .md file found on {folder}")

    return md_files[0]


def main():
    languages_path = BASE_DIR / "languages.json"

    with open(languages_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    languages = data["languages"]

    source_dir = BASE_DIR / SOURCE_LANG

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    source_md = get_single_md_file(source_dir)

    source_content = source_md.read_text(encoding="utf-8")

    print(f"Base file found: {source_md.name}")

    for lang in languages:
        code = lang["code"]
        name = lang["name"]

        if code in SKIP_LANGS:
            continue

        print(f"\n=== Processing {code} ({name}) ===")

        target_dir = BASE_DIR / code
        target_dir.mkdir(parents=True, exist_ok=True)

        translated_filename = translate_filename(
            source_md.name,
            name
        )

        translated_content = translate_questions(
            source_content,
            name
        )

        output_path = target_dir / translated_filename
        output_path.write_text(
            translated_content,
            encoding="utf-8"
        )

        print(f"Saved: {output_path}")

        # Prevents Rate Limit.
        time.sleep(2)


if __name__ == "__main__":
    main()