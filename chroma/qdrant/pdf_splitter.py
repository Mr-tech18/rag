from pypdf import PdfReader , PdfWriter


reader =PdfReader('./data/ilovepdf_merged.pdf')

writer=PdfWriter()

for page in reader.pages[:2]:
        writer.add_page(page)

writer.write('./data/ilovepdf_merged-1-2.pdf')