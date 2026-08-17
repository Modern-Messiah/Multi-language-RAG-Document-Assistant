"""Encoding handling in DocumentLoader.load_txt."""
from app.rag.document_loader import DocumentLoader


def test_load_txt_cp1251_decodes_cyrillic(tmp_path):
    text = "Привет, мир! Это тест кодировки для русского текста."
    path = tmp_path / "ru_cp1251.txt"
    path.write_bytes(text.encode("cp1251"))

    docs = DocumentLoader().load_document(str(path))

    assert "Привет" in docs[0].page_content
    assert "кодировки" in docs[0].page_content


def test_load_txt_utf8_bom_strips_bom(tmp_path):
    text = "Simple UTF-8 text with BOM marker."
    path = tmp_path / "bom.txt"
    path.write_bytes(text.encode("utf-8-sig"))

    docs = DocumentLoader().load_document(str(path))

    assert not docs[0].page_content.startswith("﻿")
    assert docs[0].page_content.strip() == text
