from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SinhalaChatbot:
    def __init__(self):
        # Load translation models
        self.translation_model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt")
        self.translation_tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt")
        
        # Load conversation model (using a smaller model as an example)
        self.conversation_model = AutoModelForCausalLM.from_pretrained(
            "gpt2-medium")  # You might want to use a more powerful model
        self.conversation_tokenizer = AutoTokenizer.from_pretrained(
            "gpt2-medium")
        
    def translate_to_english(self, sinhala_text):
        # Set source language to Sinhala
        self.translation_tokenizer.src_lang = "si_LK"
        
        # Encode the Sinhala text
        inputs = self.translation_tokenizer(sinhala_text, return_tensors="pt", padding=True)
        
        # Set forced language for output
        forced_bos_token_id = self.translation_tokenizer.lang_code_to_id["en_XX"]
        
        # Generate English translation
        translated_ids = self.translation_model.generate(
            inputs["input_ids"], 
            forced_bos_token_id=forced_bos_token_id,
            max_length=100
        )
        
        # Decode the translation
        english_text = self.translation_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        return english_text
    
    def translate_to_sinhala(self, english_text):
        # Set source language to English
        self.translation_tokenizer.src_lang = "en_XX"
        
        # Encode the English text
        inputs = self.translation_tokenizer(english_text, return_tensors="pt", padding=True)
        
        # Set forced language for output
        forced_bos_token_id = self.translation_tokenizer.lang_code_to_id["si_LK"]
        
        # Generate Sinhala translation
        translated_ids = self.translation_model.generate(
            inputs["input_ids"], 
            forced_bos_token_id=forced_bos_token_id,
            max_length=100
        )
        
        # Decode the translation
        sinhala_text = self.translation_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        return sinhala_text
    
    def generate_response(self, english_query):
        # Add a prompt to help guide the model's response
        prompt = f"User: {english_query}\nAssistant:"
        
        # Encode the prompt
        inputs = self.conversation_tokenizer(prompt, return_tensors="pt")
        
        # Generate a response
        response_ids = self.conversation_model.generate(
            inputs["input_ids"],
            max_length=150,
            temperature=0.7,
            pad_token_id=self.conversation_tokenizer.eos_token_id
        )
        
        # Decode the response
        full_response = self.conversation_tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        assistant_response = full_response.split("Assistant:")[1].strip()
        return assistant_response
    
    def chat(self, user_input_sinhala):
        # Step 1: Translate Sinhala input to English
        english_query = self.translate_to_english(user_input_sinhala)
        
        # Step 2: Generate English response
        english_response = self.generate_response(english_query)
        
        # Step 3: Translate English response back to Sinhala
        sinhala_response = self.translate_to_sinhala(english_response)
        
        return sinhala_response

# Create the Sinhala chatbot object
chatbot = SinhalaChatbot()

# Start the conversation
print("ආයුබෝවන්! මට ඔබට උදව් කරන්න පුළුවන්ද?")  # "Hello! Can I help you?"
while True:
    user_input = input("ඔබ: ")  # "You: "
    
    if user_input.lower() == 'exit':
        print("ඔබ හමුවීම සතුටක්! සුභ දවසක්!")  # "Nice meeting you! Have a good day!"
        break
    
    # Get the chatbot's response
    response = chatbot.chat(user_input)
    print("චැට්බොට්: " + response)  # "Chatbot: "