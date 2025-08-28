#!/usr/bin/env python3

import sys
sys.path.append('src')

from prompts.llm_explainer import DogSpeakExplainer, ExplanationRequest

def translate_dog_vocalization(audio_analysis: str) -> str:
    """Translate dog vocalization analysis into plain language"""
    
    # Parse the audio analysis input
    lines = audio_analysis.strip().split('\n')
    
    # Extract intents
    intents_line = [line for line in lines if 'Intents:' in line][0]
    intent_parts = intents_line.split('Intents:')[1].strip().split(',')
    
    # Parse primary intent and confidence
    primary_intent_str = intent_parts[0].strip()
    intent_name = primary_intent_str.split('(')[0].strip()
    confidence_str = primary_intent_str.split('(')[1].replace(')', '')
    confidence = float(confidence_str)
    
    # Map display names to internal names
    intent_mapping = {
        'Alarm/Guard': 'alarm_guard',
        'Playful': 'play_invitation',
        'Territorial': 'territorial',
        'Distress': 'distress_separation',
        'Attention': 'attention_seeking'
    }
    
    tier1_intent = intent_mapping.get(intent_name, 'alarm_guard')
    
    # Extract context tags
    context_line = [line for line in lines if 'Context tags:' in line]
    tier2_tags = []
    if context_line:
        tags_str = context_line[0].split('Context tags:')[1].strip()
        tier2_tags = [tag.strip() for tag in tags_str.split(',')]
    
    # Extract metadata
    metadata_line = [line for line in lines if 'Metadata:' in line][0]
    metadata_str = metadata_line.split('Metadata:')[1].strip()
    
    # Parse metadata
    breed = None
    environment = None
    duration = None
    
    for item in metadata_str.split(','):
        key_value = item.strip().split('=')
        if len(key_value) == 2:
            key, value = key_value
            if key == 'breed':
                breed = value
            elif key == 'environment':
                environment = value
            elif key == 'duration':
                duration = float(value.replace('s', ''))
    
    # Create explanation request
    request = ExplanationRequest(
        tier1_intent=tier1_intent,
        tier1_confidence=confidence,
        tier2_tags=tier2_tags,
        tier2_probs={tag: 0.8 for tag in tier2_tags},
        overall_confidence=confidence,
        metadata={'duration': duration} if duration else {},
        breed=breed,
        environment=environment
    )
    
    # Generate explanation
    explainer = DogSpeakExplainer()
    result = explainer.generate_explanation(request)
    
    # Format output
    output = f"{result['explanation']}\nTips:\n1. {result['advice']}"
    
    # Add second tip based on context
    if 'doorbell' in tier2_tags and 'stranger' in tier2_tags:
        if tier1_intent == 'alarm_guard':
            output += "\n2. Reward her once you confirm everything is safe, to reinforce controlled guarding."
        else:
            output += "\n2. Acknowledge their alert and provide reassurance once the situation is assessed."
    else:
        output += "\n2. Monitor their body language for additional context clues."
    
    return output

def main():
    """Test the translation function"""
    
    # Test with the provided example
    test_input = """Audio Analysis:
- Intents: Alarm/Guard (0.81), Playful (0.09)
- Context tags: doorbell, stranger
- Metadata: breed=German Shepherd, environment=indoor, snr=22dB, duration=5.3s"""
    
    print("🐕 DogSpeak Live Translation")
    print("=" * 40)
    print("Input:")
    print(test_input)
    print("\nOutput:")
    
    translation = translate_dog_vocalization(test_input)
    print(translation)
    
    # Test with another example
    test_input2 = """Audio Analysis:
- Intents: Playful (0.92), Attention (0.05)
- Context tags: toy_present, high_energy
- Metadata: breed=Labrador, environment=outdoor, duration=2.1s"""
    
    print("\n" + "="*40)
    print("Input:")
    print(test_input2)
    print("\nOutput:")
    
    translation2 = translate_dog_vocalization(test_input2)
    print(translation2)

if __name__ == "__main__":
    main()
