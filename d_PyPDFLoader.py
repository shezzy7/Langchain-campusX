""" PyPDFLoader is a document loader in langchain used to load content from PDF files and convert each page into a Document object

Example Output ->
[
  Document(
    page_content="This is the text from page 1...",
    metadata={
      'source': 'example.pdf',
      'page': 0
    }
  ),
  Document(
    page_content="This is the text from page 2...",
    metadata={
      'source': 'example.pdf',
      'page': 1
    }
  ),
  ...
]

 Limitations- > Is uses the PyPDF library under the hood - not great for scanned PDFs of complex layouts.
"""

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('pdflatex-outline.pdf')

data = loader.load()

print(data[1].page_content)