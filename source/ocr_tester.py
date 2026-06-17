import easyocr

reader = easyocr.Reader(["en"], gpu=False)

results = reader.readtext("CML.jpg", detail=0)

for line in results:
    print(line)