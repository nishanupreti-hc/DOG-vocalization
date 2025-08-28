#!/usr/bin/env python3

import sys
sys.path.append('src')

from translation.dog_translator import DogTranslator

def interactive_demo():
    """Interactive demo of dog translation without audio"""
    
    translator = DogTranslator()
    
    print("🐕➡️📝 Dog-to-English Translator Demo")
    print("=" * 40)
    print("Available vocalizations: bark, whine, growl, howl, whimper")
    print("Type 'quit' to exit\n")
    
    while True:
        vocalization = input("🐕 What sound did your dog make? ").lower().strip()
        
        if vocalization == 'quit':
            print("👋 Goodbye!")
            break
        
        if vocalization not in ['bark', 'whine', 'growl', 'howl', 'whimper']:
            print("❌ Unknown vocalization. Try: bark, whine, growl, howl, or whimper")
            continue
        
        # Get translation
        translation = translator.translate(vocalization)
        
        print(f"\n🗣️  Translation: \"{translation['translation']}\"")
        print(f"💭 Emotion: {translation['emotion']}")
        
        # Get behavioral advice
        advice = translator.get_behavioral_advice(vocalization, 'default')
        print(f"💡 Advice: {advice}")
        print("-" * 40)

def example_translations():
    """Show example translations"""
    
    translator = DogTranslator()
    
    print("🐕➡️📝 Example Dog Translations")
    print("=" * 40)
    
    examples = [
        ("bark", "High-pitched rapid barking"),
        ("whine", "Soft whining sound"),
        ("growl", "Low rumbling growl"),
        ("howl", "Long sustained howl"),
        ("whimper", "Quiet whimpering")
    ]
    
    for vocalization, description in examples:
        translation = translator.translate(vocalization)
        
        print(f"\n🔊 {description}")
        print(f"🗣️  \"{translation['translation']}\"")
        print(f"💭 {translation['emotion']}")
        
        advice = translator.get_behavioral_advice(vocalization, 'default')
        print(f"💡 {advice}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'examples':
        example_translations()
    else:
        interactive_demo()

if __name__ == "__main__":
    main()
