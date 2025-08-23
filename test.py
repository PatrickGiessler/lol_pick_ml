import easyocr
reader = easyocr.Reader(['de','en','es']) # this needs to run only once to load the model into memory
result = reader.readtext('pick1.png')
print(result)