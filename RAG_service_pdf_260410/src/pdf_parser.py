import pdfplumber

def parse_pdf(pdf_path: str) -> list:
    """PDF에서 페이지별 텍스트 추출 (pdfplumber 기반)"""
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({
                "page": page_num + 1,
                "text": text.strip()
            })

    return pages

# 표(Table) 추출 기능 확장
def extract_tables(pdf_path: str):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            extracted = page.extract_tables()
            for table in extracted:
                tables.append({
                    "page": page_num + 1,
                    "table": table
                })
    return tables