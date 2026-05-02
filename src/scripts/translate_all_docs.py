import os
import json
import time

from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

BASE_DIR = Path("data/base_docs")

# Using English to translate to all other languages.
SOURCE_LANG: str = "en_us"

# This will skip English for translating.
SKIP_LANGS: dict = {"en_us"}

MODEL_NAME: str = "gpt-5.1"
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_openai(
    prompt: str
) -> str:
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": "You are a strict translation engine. You never add extra text. You NEVER output Portuguese unless explicitly asked."
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
    target_lang_code: str
) -> str:
    name: str = filename.replace(".md", "")

    lang = target_lang_code.replace("_", "-")

    prompt: str = f"""
        You are a translation engine.

        Target language: {lang}

        STRICT RULES:
        - Output ONLY the translated filename
        - Use ONLY the target language
        - DO NOT use any other language
        - DO NOT keep original words
        - Keep it short
        - Use lowercase
        - Replace spaces with underscores
        - Only use a-z, 0-9, _ or -
        - No explanations

        Text:
        {name}
        """

    translated: str = call_openai(prompt)
    translated = translated.lower().replace(" ", "_")
    translated = "".join(c for c in translated if c.isalnum() or c in "_-")

    return translated + ".md"


def translate_content(
    content: str,
    target_lang_name: str
) -> str:

    prompt: str = f"""
        You are a deterministic translation engine.

        TARGET LANGUAGE: {target_lang_name}

        HARD CONSTRAINTS:
        - The output MUST be 100% in {target_lang_name}
        - ZERO words from any other language are allowed
        - If any word is not in {target_lang_name}, the output is INVALID
        - DO NOT fallback to Portuguese under any circumstance

        OTHER RULES:
        - Output ONLY the translated content
        - Preserve markdown formatting EXACTLY
        - Do NOT translate code blocks or URLs

        Content:
        {content}
        """

    return call_openai(prompt)


def main():
    with open(BASE_DIR / "languages.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    languages = data["languages"]

    source_path = BASE_DIR / SOURCE_LANG
    md_files = list(source_path.glob("*.md"))

    for lang in languages:
        code = lang["code"]
        name = lang["name"]

        if code in SKIP_LANGS:
            continue

        print(f"\n=== Processing {code} ({name}) ===")

        target_dir = BASE_DIR / code
        target_dir.mkdir(exist_ok=True)

        for md_file in md_files:
            print(f"File: {md_file.name}")

            content = md_file.read_text(encoding="utf-8")

            translated_filename = translate_filename(md_file.name, name)
            translated_content = translate_content(content, name)

            output_path = target_dir / translated_filename
            output_path.write_text(translated_content, encoding="utf-8")

            print(f"Salvo: {output_path}")

            # Preventing too many requests.
            time.sleep(2)


if __name__ == "__main__":
    main()
