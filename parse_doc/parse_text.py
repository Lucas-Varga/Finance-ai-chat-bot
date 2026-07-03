import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_unstructured import UnstructuredLoader

BASE_DIR = Path("/Users/vargalb/Documents/Automações/FinanceChatBot/base")

def group_elements_by_page(elements: List[Any]) -> Dict[int, str]:
    pages: Dict[int, List[str]] = {}

    for element in elements:
        metadata = getattr(element, "metadata", {}) or {}
        page_number = metadata.get("page_number", 1)

        text = getattr(element, "page_content", None)
        
        if text is None and hasattr(element, "text"):
            text = getattr(element, "text")
        if text is None and isinstance(element, dict):
            text = element.get("text", element.get("page_content", ""))
        if text is None:
            text = str(element)

        pages.setdefault(page_number, []).append(text)

    return {page: "\n".join(parts).strip() for page, parts in pages.items()}


def build_page_json_objects(pages: Dict[int, str], source: str) -> List[Dict[str, Any]]:
    return [
        {
            "page_content": page_text,
            "metadata": {
                "source": source,
                "page": page_number
            }
        }
        for page_number, page_text in sorted(pages.items())
    ]


def extract_pages_as_json(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    loader = UnstructuredLoader(str(path))
    elements = loader.load()

    pages = group_elements_by_page(elements)
    return build_page_json_objects(pages, source=str(path))


def save_pages_as_json(file_path: str, output_path: str) -> None:
    page_objects = extract_pages_as_json(file_path)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(page_objects, output_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_file = BASE_DIR / "doc1.pdf"
    output_file = Path("/Users/vargalb/Documents/Automações/FinanceChatBot/output/resultado.json")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    save_pages_as_json(str(input_file), str(output_file))
    print(f"Páginas extraídas e salvas em: {output_file}")