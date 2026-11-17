from marsrag.ingestion.chunker import stable_chunk_id,chunk_pages
from marsrag.ingestion.models import PageText

def test_id_is_deterministic_and_page_sensitive():
    value="Mars transit duration depends on mission design."
    assert stable_chunk_id("RPT",2,value)==stable_chunk_id("RPT",2,value)
    assert stable_chunk_id("RPT",2,value)!=stable_chunk_id("RPT",3,value)

def test_chunk_page_and_section_metadata():
    page=PageText(7,"1 Mission Overview\\n"+"Mars missions require planning and support systems. "*12)
    chunks=chunk_pages([page],report_no="RPT",title="Test",year=2024,url="https://example.org",status="current")
    assert chunks and all(c.page==7 for c in chunks)
