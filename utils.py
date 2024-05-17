from langdetect import detect
from deep_translator import GoogleTranslator

class Translation:
    def __init__(self, text, destination):
        self.text = text
        self.destination = destination
        try:
            self.original = detect(self.text)
        except Exception as e:
            self.original = "auto"
    def translatef(self):
        translator = GoogleTranslator(source=self.original, target=self.destination)
        translation = translator.translate(self.text)
        return translation
    
class NeuralSearcher:
    def __init__(self, collection, encoder):
        self.collection = collection
        self.encoder = encoder
    def search(self, text):
        results = self.collection.query(
            data=self.encoder.encode(text).tolist(),  # required
            limit=1,                     # number of records to return
            filters={},                  # metadata filters
            measure="cosine_distance",   # distance measure to use
            include_value=True,         # should distance measure values be returned?
            include_metadata=True,      # should record metadata be returned?
        )
        return results