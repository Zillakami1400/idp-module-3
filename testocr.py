from ocr.processor import process_document

doc_id = "test_doc"

file_path = "dataset/invoices/sample_invoice.pdf"

text = process_document(doc_id, file_path)

print(text)