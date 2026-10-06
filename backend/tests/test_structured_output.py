import pytest

from app.llm.exceptions import LLMStructuredOutputError
from app.llm.structured import extract_json_text, parse_structured_output
from app.models.research_report import SynthesizerLLMOutput


def test_extract_json_from_markdown_fence():
    raw = 'Here you go:\n```json\n{"executive_summary": "hi"}\n```'
    assert extract_json_text(raw) == '{"executive_summary": "hi"}'


def test_extract_json_after_prose():
    raw = 'We need to produce a structured report.\n\n{"executive_summary": "Done", "key_findings": []}'
    text = extract_json_text(raw)
    parsed = parse_structured_output(text, SynthesizerLLMOutput)
    assert parsed.executive_summary == "Done"


def test_parse_fails_without_json():
    with pytest.raises(LLMStructuredOutputError):
        parse_structured_output("We need to produce a str...", SynthesizerLLMOutput)
