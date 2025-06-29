from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class FalconModel:
    def __init__(self):
        self.model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.generator = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def get_response(self, prompt):
        result = self.generator(prompt, max_length=100, do_sample=True)[0]["generated_text"]
        return result