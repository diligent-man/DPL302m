import json
class Model_en:
    def __init__(self) -> None:
        self.weight_path = ""
        self.model = None
        self.isload = False
    
    def load_model(self):
        self.isload = True
    
    def isload(self):
        return self.isload
    def predict(self, input_data):
        assert self.isload == True, Exception("The model have not been load")
        return "Hello world"
