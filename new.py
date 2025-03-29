from deep_translator import GoogleTranslator

def translate_text(text, dest_language):
   
    translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
    return translated

if __name__ == "__main__":
  
    text_to_translate = input("Enter the text you want to translate: ")
    target_language = input("Enter the target language code (e.g., 'es' for Spanish, 'fr' for French): ")
  
    translated_text = translate_text(text_to_translate, target_language)
    
  
    print(f"Translated text: {translated_text}")