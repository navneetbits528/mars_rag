import hashlib, re
from marsrag.ingestion.models import PageText,ChunkRecord
from marsrag.settings import get_settings

_NUM=re.compile(r"^\s*(\d+(?:\.\d+){0,4})\s+([A-Z][^\n]{2,120})$")
_ROMAN = re.compile(
    r"^\s*([IVXLCDM]{1,8})\.\s+([A-Z][A-Z0-9 /,:()&\-]{2,120})\s*$"
)
_CAPS=re.compile(r"^\s*([A-Z][A-Z0-9 /,:()\-]{4,100})\s*$")

def normalize_text(text: str) -> str:
    return re.sub(r"\s+"," ",text).strip()

def stable_chunk_id(report_no: str,page: int,text: str) -> str:
    raw=f"{report_no}|{page}|{normalize_text(text).casefold()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def _heading(line):
    match = _NUM.match(line)
    if match:
        return match.group(1), match.group(2).strip()

    match = _ROMAN.match(line)
    if match:
        return match.group(1), match.group(2).strip()

    match = _CAPS.match(line)
    if match and not line.isdigit():
        return "", match.group(1).title()

    return None

def _split(text,size,overlap):
    text=text.strip()
    if not text: return []
    if len(text)<=size: return [text]
    cut=text.rfind("\n\n",0,size)
    if cut<size//2: cut=text.rfind(". ",0,size)
    if cut<size//2: cut=text.rfind(" ",0,size)
    if cut<size//3: cut=size
    part=text[:cut].strip()
    if not part: part,cut=text[:size].strip(),size
    return [part]+_split(text[max(0,cut-overlap):],size,overlap)

def chunk_pages(pages: list[PageText],*,report_no,title,year,url,status) -> list[ChunkRecord]:
    s=get_settings()
    size=s.chunk_size_chars
    overlap=int(size*s.chunk_overlap_ratio)
    result=[]
    stack=[]
    for page in pages:
        paragraphs=[]
        def flush():
            nonlocal paragraphs
            body="\n".join(paragraphs).strip()
            paragraphs=[]
            if len(body)<s.min_chunk_chars: return
            section=stack[-1] if stack else "Introduction"
            path=" > ".join(stack) if stack else section
            for part in _split(body,size,overlap):
                if len(part)<s.min_chunk_chars: continue
                normalized=normalize_text(part)
                cid=stable_chunk_id(report_no,page.page_number,normalized)
                result.append(ChunkRecord(cid,report_no,page.page_number,title,int(year),status,url,
                    section,path,"text",part,f"Section: {section}\n{part}",
                    hashlib.sha256(normalized.encode()).hexdigest(),
                    {"section_title":section,"section_path":path,"ocr":bool(page.metadata.get("ocr"))}))
        for line in page.text.splitlines():
            heading=_heading(line.strip())
            if heading:
                flush()
                number,name=heading
                if number:
                    depth=number.count(".")+1
                    stack[:]=stack[:depth-1]
                    stack.append(f"{number} {name}")
                else:
                    stack[:]=[name]
            else:
                paragraphs.append(line)
        flush()
    unique={c.chunk_id:c for c in result}
    return list(unique.values())
