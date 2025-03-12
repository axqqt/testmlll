import json
import logging
import time
from functools import lru_cache
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SinhalaChatbot:
    def __init__(self, device=None, model_cache_size=5, knowledge_base_path="company_data.json"):
        # Initialize device (CUDA if available, otherwise CPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load models with error handling
        try:
            logger.info("Loading translation model...")
            self.translation_model = MBartForConditionalGeneration.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt").to(self.device)
            self.translation_tokenizer = MBart50TokenizerFast.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt")
            
            logger.info("Loading conversation model...")
            self.conversation_model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-neo-1.3B").to(self.device)
            self.conversation_tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neo-1.3B")
            
            # Set pad token for the conversation model if not already set
            if self.conversation_tokenizer.pad_token is None:
                self.conversation_tokenizer.pad_token = self.conversation_tokenizer.eos_token
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
        
        # Batch processing parameters
        self.batch_size = 4
        self.lru_cache_size = model_cache_size
        
        # Enable half-precision for better performance on CUDA devices
        if self.device == 'cuda' and torch.cuda.is_available():
            self.translation_model = self.translation_model.half()
            self.conversation_model = self.conversation_model.half()
        
        # Chat history for context
        self.chat_history = []
        self.max_history_length = 5
        
        # Load knowledge base
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)

    def load_knowledge_base(self, file_path):
        """Load company-specific data from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            logger.warning(f"Knowledge base file not found: {file_path}. Creating empty knowledge base.")
            # Create an empty file with basic structure to avoid future FileNotFoundError
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump({}, file)
            return {}
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            return {}

    def search_knowledge_base(self, query):
        """Search the knowledge base for a matching response."""
        query_lower = query.lower()
        for question, answer in self.knowledge_base.items():
            if question.lower() in query_lower:
                return answer
        return None

    @lru_cache(maxsize=128)
    def translate_to_english(self, sinhala_text):
        """Translate Sinhala text to English with caching for repeated queries."""
        start_time = time.time()
        try:
            # Set source language to Sinhala
            self.translation_tokenizer.src_lang = "si_LK"
            inputs = self.translation_tokenizer(
                sinhala_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            # Set forced language for output
            forced_bos_token_id = self.translation_tokenizer.lang_code_to_id["en_XX"]
            with torch.no_grad():
                translated_ids = self.translation_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True
                )
            english_text = self.translation_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            logger.debug(f"Translation to English took {time.time() - start_time:.2f}s")
            return english_text
        except Exception as e:
            logger.error(f"Error translating to English: {str(e)}")
            return f"Translation error: {str(e)}"

    @lru_cache(maxsize=128)
    def translate_to_sinhala(self, english_text):
        """Translate English text to Sinhala with caching for repeated responses."""
        start_time = time.time()
        try:
            # Set source language to English
            self.translation_tokenizer.src_lang = "en_XX"
            inputs = self.translation_tokenizer(
                english_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            # Set forced language for output
            forced_bos_token_id = self.translation_tokenizer.lang_code_to_id["si_LK"]
            with torch.no_grad():
                translated_ids = self.translation_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True
                )
            sinhala_text = self.translation_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            logger.debug(f"Translation to Sinhala took {time.time() - start_time:.2f}s")
            return sinhala_text
        except Exception as e:
            logger.error(f"Error translating to Sinhala: {str(e)}")
            return f"පරිවර්තන දෝෂයකි: {str(e)}"

    def generate_response(self, english_query):
        """Generate a contextual response based on chat history and knowledge base."""
        try:
            # Step 1: Check knowledge base for a direct match
            kb_response = self.search_knowledge_base(english_query)
            if kb_response:
                logger.info("Found response in knowledge base.")
                return kb_response
            
            # Step 2: Fall back to conversational model
            context = ""
            for i, (q, a) in enumerate(self.chat_history[-self.max_history_length:]):
                context += f"User: {q}\nAssistant: {a}\n"
            prompt = f"{context}User: {english_query}\nAssistant:"
            
            inputs = self.conversation_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                response_ids = self.conversation_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_length=1024,
                    min_length=10,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.conversation_tokenizer.eos_token_id
                )
            
            full_response = self.conversation_tokenizer.decode(response_ids[0], skip_special_tokens=True)
            
            # Extract the assistant's response from the generated text
            if "Assistant:" in full_response:
                parts = full_response.split("Assistant:")
                assistant_response = parts[-1].strip()
            else:
                # If "Assistant:" is not found, try to extract after the last "User:"
                parts = full_response.split("User:")
                if len(parts) > 1:
                    assistant_response = parts[-1].strip()
                else:
                    assistant_response = full_response.strip()
            
            if not assistant_response:
                assistant_response = "I'm not sure how to respond to that."
            
            return assistant_response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def chat(self, user_input):
        """Process a single turn of chat with improved error handling."""
        try:
            # Check if input is already in Sinhala
            is_sinhala = self.is_sinhala_text(user_input)
            
            # Step 1: If Sinhala, translate to English; otherwise, use as is
            if is_sinhala:
                english_query = self.translate_to_english(user_input)
                logger.info(f"Translated query from Sinhala: {english_query}")
            else:
                english_query = user_input
                logger.info(f"Using English query directly: {english_query}")
            
            # Step 2: Generate English response
            english_response = self.generate_response(english_query)
            logger.info(f"Generated response: {english_response}")
            
            # Step 3: If original input was Sinhala, translate response back to Sinhala
            if is_sinhala:
                final_response = self.translate_to_sinhala(english_response)
            else:
                final_response = english_response
            
            # Update chat history for context in future turns
            self.chat_history.append((english_query, english_response))
            if len(self.chat_history) > self.max_history_length:
                self.chat_history.pop(0)
            
            return final_response
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            error_msg = "දෝෂයක් සිදු විය. කරුණාකර නැවත උත්සාහ කරන්න." if self.is_sinhala_text(user_input) else "An error occurred. Please try again."
            return error_msg
    
    def is_sinhala_text(self, text):
        """Detect if text contains Sinhala characters."""
        # Sinhala Unicode range is U+0D80 to U+0DFF
        for char in text:
            if '\u0D80' <= char <= '\u0DFF':
                return True
        return False

    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        logger.info("Chat history cleared")

    def update_knowledge_base(self, question, answer, save_to_file=True):
        """Add or update an entry in the knowledge base."""
        try:
            self.knowledge_base[question] = answer
            if save_to_file:
                with open("company_data.json", "w", encoding="utf-8") as file:
                    json.dump(self.knowledge_base, file, ensure_ascii=False, indent=2)
            logger.info(f"Knowledge base updated with question: {question}")
            return True
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")
            return False


def run_chatbot():
    print("\n" + "="*50)
    print("  සිංහල චැට්බොට් - Sinhala Chatbot")
    print("="*50)
    try:
        chatbot = SinhalaChatbot()
        print("\nආයුබෝවන්! මට ඔබට උදව් කරන්න පුළුවන්ද?")
        print("\nවිශේෂ විධාන / Special commands:")
        print("  'exit' - නවත්තන්න / Exit")
        print("  'clear' - සංවාද ඉතිහාසය මකන්න / Clear chat history")
        print("-"*50)
        while True:
            user_input = input("\nඔබ: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nචැට්බොට්: ඔබ හමුවීම සතුටක්! සුභ දවසක්!")
                break
            elif user_input.lower() in ['clear', 'reset']:
                chatbot.clear_history()
                print("\nචැට්බොට්: ඉතිහාසය හිස් කර ඇත. අපට අලුත් සංවාදයක් ආරම්භ කළ හැකිය.")
                continue
            start_time = time.time()
            response = chatbot.chat(user_input)
            processing_time = time.time() - start_time
            print(f"\nචැට්බොට්: {response}")
            logger.debug(f"Processing time: {processing_time:.2f}s")
    except KeyboardInterrupt:
        print("\nවැඩසටහන අවසන් කරයි. / Program terminated.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("දෝෂයක් සිදු විය. වැඩසටහන අවසන් කරයි. / An error occurred. Program terminated.")


if __name__ == "__main__":
    run_chatbot()