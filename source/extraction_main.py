from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests
import json
import os
import re
from pdf2image import convert_from_path
import easyocr

# Create OCR model ONCE
ocr_reader = easyocr.Reader(["en"], gpu=False)

@tool
def save_json(filename: str, data: str) -> str:
    """Save valid JSON data to a file."""

    if not filename.endswith(".json"):
        filename += ".json"

    try:
        parsed_data = json.loads(data)
    except json.JSONDecodeError:
        return "Error: data was not valid JSON, so it was not saved."

    with open(filename, "w") as f:
        json.dump(parsed_data, f, indent=4)

    return f"Saved valid JSON to {filename}"


def normalize_json_text(text: str) -> str:
    """Normalize text to make JSON detection more reliable.
    This function converts fancy Unicode characters into normal JSON-friendly characters."""

    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    if not isinstance(text, str):
        text = json.dumps(text)

    replacements = {
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '–': '-',
        '—': '-',
        '…': '...',
        '（': '(',
        '）': ')',
        '［': '[',
        '］': ']',
        '｛': '{',
        '｝': '}',
        '\u00a0': ' ',
        '\u200b': '',
        '\u2013': '-',
        '\u2014': '-',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def sanitize_json_text(text: str) -> str:
    """Escape invalid characters inside JSON string literals."""

    result = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            result.append(ch)
            escape = False
            continue

        if ch == "\\":
            result.append(ch)
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue

        if in_string:
            if ch == "\n":
                result.append("\\n")
                continue
            if ch == "\r":
                result.append("\\r")
                continue
            if ch == "\t":
                result.append("\\t")
                continue
            if ord(ch) < 0x20:
                result.append(f"\\u{ord(ch):04x}")
                continue

        result.append(ch)

    return ''.join(result)


def find_json_candidate(text: str) -> str | None:
    """Find the first balanced JSON object or array in the text."""
    for i, ch in enumerate(text):
        if ch not in '{[':
            continue

        opener = ch
        closer = '}' if ch == '{' else ']'
        depth = 0
        in_string = False
        escape = False

        for j in range(i, len(text)):
            ch2 = text[j]

            if escape:
                escape = False
                continue

            if ch2 == "\\":
                escape = True
                continue

            if ch2 == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch2 == opener:
                depth += 1
            elif ch2 == closer:
                depth -= 1
                if depth == 0:
                    return text[i:j + 1]

    return None


def extract_json(text):
    """Extract JSON object or array from model output."""

    text = normalize_json_text(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    json_text = find_json_candidate(text)
    if json_text is None:
        # Retry on cleaned version for common unicode braces or quotes
        cleaned = normalize_json_text(text)
        json_text = find_json_candidate(cleaned)

    if json_text is None:
        raise ValueError(
            "No JSON object found. "
            f"Response preview: {repr(text[:300])}"
        )

    json_text = sanitize_json_text(json_text)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        trailing_commas_fixed = re.sub(r',\s*(?=[}\]])', '', json_text)
        if trailing_commas_fixed != json_text:
            try:
                return json.loads(trailing_commas_fixed)
            except json.JSONDecodeError:
                pass

        snippet_start = max(0, e.pos - 40)
        snippet_end = min(len(json_text), e.pos + 40)
        snippet = json_text[snippet_start:snippet_end]
        raise ValueError(
            f"Invalid JSON: {e.msg} at position {e.pos}. "
            f"Snippet: {snippet!r}"
        ) from e

@tool
def get_data(path: str) -> str:
    """Read PDF using both raw text extraction and OCR, then combine the results."""

    if not os.path.exists(path):
        return f"File not found: {path}. Available files: {os.listdir()}"

    if os.path.getsize(path) == 0:
        return f"File exists but is empty: {path}"

    raw_text = ""
    ocr_text = ""

    # 1. Raw PDF text extraction
    try:
        reader = PdfReader(path)

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                raw_text += f"\n--- RAW TEXT PAGE {i + 1} ---\n"
                raw_text += page_text + "\n"

    except Exception as e:
        raw_text = f"\nRaw PDF extraction failed: {str(e)}\n"

    # 2. OCR extraction from PDF pages
    try:
        images = convert_from_path(path, dpi=200)

        for i, image in enumerate(images):
            results = ocr_reader.readtext(image, detail=0)
            page_ocr_text = "\n".join(results)

            if page_ocr_text.strip():
                ocr_text += f"\n--- OCR TEXT PAGE {i + 1} ---\n"
                ocr_text += page_ocr_text + "\n"

    except Exception as e:
        ocr_text = f"\nOCR extraction failed: {str(e)}\n"

    combined_text = f"""
RAW PDF TEXT:
{raw_text}

OCR TEXT:
{ocr_text}
"""

    return combined_text[:12000]

@tool
def list_files() -> str:
    """List all files available in the current working directory. Use this tool whenever the user asks what files exist, what files are available, or asks to list files."""

    files = []

    for file in os.listdir():
        if os.path.isfile(file):
            files.append(f"{file} - {os.path.getsize(file)} bytes")

    return "\n".join(files)

# Local Ollama model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


agent = create_agent(
    model=llm,
    tools=[
        get_data,
        save_json,
        list_files
    ],
    system_prompt="""
You are a helpful assistant.

If the user provides a PDF filename and asks you to extract information:
1. Use get_data to read the PDF.
2. Return ONLY valid JSON.
3. Do not use markdown.
4. Do not include explanations.
5. Do not wrap the JSON in ```json.
6. The JSON must use key-value format.

Only save text if it is in English Langauge

Examples of saving key-value pair:
{
"Beneficiary Name": "Anjini Jamwal",
"Age": 18
}


For normal conversation, answer normally.

The file that will be entered by the user will be present under the PDFs folder.

Files are mostly going to be of Client Master List Type, 
which means you MUST focus on getting client details accurately.
""")


while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    # If user gives a PDF filename, handle it reliably with Python
    if user_input.lower().endswith(".pdf"):
        text = get_data.invoke({"path": "../PDFs/" + user_input})

        extraction_prompt = f"""
Extract structured data from the following PDF text.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanation.
Return only the JSON object with no text before or after.

PDF text:
{text}
"""

        response = llm.invoke(extraction_prompt)
        content = response.content

        print("\nAgent:", content)

        try:
            parsed = extract_json(content)

            os.makedirs("../ExtractedJSon", exist_ok=True)
            output_name = os.path.basename(user_input).replace(".pdf", ".json")
            output_path = os.path.join("../ExtractedJSon", output_name)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=4)

            print("\nSaved to " + output_path)

        except ValueError as e:
            print(f"\nCould not save because the model did not return valid JSON: {e}")

        continue

    # Normal chatbot behavior
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    })

    final_message = result["messages"][-1]
    content = final_message.content

    if isinstance(content, list):
        content = content[0]["text"]

    print("\nAgent:", content)