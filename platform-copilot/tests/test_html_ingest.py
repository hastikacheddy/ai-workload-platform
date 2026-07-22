from platform_copilot.services.ingestion.html import parse_html

HTML = """
<html><head><title>Postmortem: API Outage</title></head>
<body>
  <h1>Postmortem: API Outage</h1>
  <p>On 2026-05-01 the API returned 503s for 20 minutes.</p>
  <h2>Root cause</h2>
  <p>A bad deploy exhausted the database connection pool.</p>
  <h2>Action items</h2>
  <ul><li>Add a pool-saturation alert.</li><li>Add a canary stage.</li></ul>
  <script>tracker()</script>
</body></html>
"""


def test_parse_html_extracts_title_and_sections() -> None:
    doc = parse_html(HTML, source_type="postmortem", source_ref="pm/api-outage.html")

    assert doc.slug == "api-outage"
    assert doc.title == "Postmortem: API Outage"
    headings = [section.heading for section in doc.sections]
    assert "Root cause" in headings
    assert "Action items" in headings
    assert "connection pool" in doc.text
    assert "tracker()" not in doc.text  # script content dropped
