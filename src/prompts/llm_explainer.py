import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import re

@dataclass
class ExplanationRequest:
    """Request structure for LLM explanation"""
    tier1_intent: str
    tier1_confidence: float
    tier2_tags: List[str]
    tier2_probs: Dict[str, float]
    overall_confidence: float
    metadata: Dict[str, any]
    breed: Optional[str] = None
    environment: Optional[str] = None

class DogSpeakExplainer:
    """Generate human-readable explanations for dog vocalizations"""
    
    def __init__(self):
        self.intent_descriptions = {
            'alarm_guard': {
                'description': 'alerting to potential threats or unusual activity',
                'typical_context': 'strangers approaching, unusual sounds, protecting territory',
                'energy_level': 'high',
                'urgency': 'high'
            },
            'territorial': {
                'description': 'defending their space or resources',
                'typical_context': 'other dogs nearby, boundary marking, resource guarding',
                'energy_level': 'medium-high',
                'urgency': 'medium'
            },
            'play_invitation': {
                'description': 'inviting interaction and play',
                'typical_context': 'toys present, owner attention, playful mood',
                'energy_level': 'high',
                'urgency': 'low'
            },
            'distress_separation': {
                'description': 'experiencing anxiety or loneliness',
                'typical_context': 'owner leaving, being alone, separation anxiety',
                'energy_level': 'medium',
                'urgency': 'medium-high'
            },
            'pain_discomfort': {
                'description': 'experiencing physical discomfort or pain',
                'typical_context': 'injury, illness, physical distress',
                'energy_level': 'low-medium',
                'urgency': 'high'
            },
            'attention_seeking': {
                'description': 'trying to get your attention',
                'typical_context': 'wanting food, walks, or general attention',
                'energy_level': 'medium',
                'urgency': 'low-medium'
            },
            'whine_appeal': {
                'description': 'making a polite request',
                'typical_context': 'needing bathroom, wanting something specific',
                'energy_level': 'low',
                'urgency': 'medium'
            },
            'growl_threat': {
                'description': 'giving a warning or feeling threatened',
                'typical_context': 'feeling cornered, resource guarding, fear',
                'energy_level': 'medium',
                'urgency': 'high'
            },
            'growl_play': {
                'description': 'engaging in playful interaction',
                'typical_context': 'tug-of-war, rough play, excitement',
                'energy_level': 'high',
                'urgency': 'low'
            },
            'howl_contact': {
                'description': 'communicating over long distances',
                'typical_context': 'responding to sirens, calling to other dogs',
                'energy_level': 'medium',
                'urgency': 'low'
            },
            'yip_puppy': {
                'description': 'expressing excitement or juvenile behavior',
                'typical_context': 'young dogs, high excitement, play',
                'energy_level': 'high',
                'urgency': 'low'
            },
            'other_unknown': {
                'description': 'expressing something unclear',
                'typical_context': 'mixed signals, unclear situation',
                'energy_level': 'medium',
                'urgency': 'medium'
            }
        }
        
        self.context_modifiers = {
            'doorbell': 'when the doorbell rang',
            'stranger': 'because a stranger is present',
            'owner_arrives': 'because you just arrived home',
            'walk_time': 'because it\'s time for a walk',
            'food_time': 'because it\'s feeding time',
            'toy_present': 'because they see their favorite toy',
            'vet': 'in the veterinary environment',
            'crate': 'related to their crate or kennel',
            'night': 'during nighttime',
            'other_dog': 'because other dogs are around',
            'thunder': 'due to thunder or storms',
            'fireworks': 'because of loud sudden noises',
            'indoor': 'while inside',
            'outdoor': 'while outside',
            'high_energy': 'in an excited state',
            'calm': 'in a relaxed mood'
        }
        
        self.actionable_advice = {
            'alarm_guard': [
                "Check what caught their attention - they're doing their job as a watchdog",
                "Acknowledge their alert, then calmly assess the situation"
            ],
            'territorial': [
                "Give them space and avoid forcing interactions",
                "Redirect their attention with a command or treat"
            ],
            'play_invitation': [
                "Engage in interactive play or exercise",
                "Use this energy for training sessions with positive reinforcement"
            ],
            'distress_separation': [
                "Provide comfort and reassurance",
                "Consider gradual desensitization to being alone"
            ],
            'pain_discomfort': [
                "Check for visible injuries or signs of illness",
                "Consult a veterinarian if the behavior persists"
            ],
            'attention_seeking': [
                "Respond to their basic needs (food, water, bathroom)",
                "Provide attention and interaction when they're calm"
            ],
            'whine_appeal': [
                "Check if they need to go outside or have other needs",
                "Respond to polite requests to reinforce good communication"
            ],
            'growl_threat': [
                "Give them space and don't force the situation",
                "Identify and remove the source of their discomfort"
            ],
            'growl_play': [
                "Join in the fun with appropriate play",
                "Ensure play stays safe and doesn't escalate"
            ],
            'howl_contact': [
                "This is normal communication behavior",
                "You can howl back to engage in 'conversation'"
            ],
            'yip_puppy': [
                "Channel their excitement into positive activities",
                "Use this energy for training and socialization"
            ],
            'other_unknown': [
                "Observe their body language for additional context",
                "Monitor the situation and respond to their basic needs"
            ]
        }
    
    def generate_explanation(self, request: ExplanationRequest) -> Dict[str, str]:
        """Generate human-readable explanation"""
        
        intent_info = self.intent_descriptions.get(request.tier1_intent, {})
        
        # Build context string
        context_parts = []
        for tag in request.tier2_tags:
            if tag in self.context_modifiers:
                context_parts.append(self.context_modifiers[tag])
        
        context_str = ', '.join(context_parts) if context_parts else 'in this situation'
        
        # Generate main explanation
        confidence_level = self._get_confidence_level(request.tier1_confidence)
        
        explanation = self._build_explanation(
            intent=request.tier1_intent,
            confidence=request.tier1_confidence,
            confidence_level=confidence_level,
            context=context_str,
            intent_info=intent_info,
            breed=request.breed
        )
        
        # Get actionable advice
        advice = self._get_advice(request.tier1_intent, request.tier2_tags)
        
        # Generate technical summary
        technical_summary = self._build_technical_summary(request)
        
        return {
            'explanation': explanation,
            'advice': advice,
            'technical_summary': technical_summary,
            'confidence_level': confidence_level
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to human-readable level"""
        if confidence >= 0.8:
            return 'Very confident'
        elif confidence >= 0.6:
            return 'Confident'
        elif confidence >= 0.4:
            return 'Somewhat confident'
        else:
            return 'Uncertain'
    
    def _build_explanation(self, intent: str, confidence: float, confidence_level: str, 
                          context: str, intent_info: Dict, breed: Optional[str]) -> str:
        """Build the main explanation text"""
        
        description = intent_info.get('description', 'communicating something')
        
        # Start with confidence and intent
        explanation = f"{confidence_level} ({confidence:.0%}) that your dog is {description}"
        
        # Add context if available
        if context != 'in this situation':
            explanation += f" {context}"
        
        # Add breed-specific note if available
        if breed and breed != 'mixed':
            breed_note = self._get_breed_note(breed, intent)
            if breed_note:
                explanation += f". {breed_note}"
        
        # Add period if not already there
        if not explanation.endswith('.'):
            explanation += '.'
        
        return explanation
    
    def _get_breed_note(self, breed: str, intent: str) -> Optional[str]:
        """Get breed-specific behavioral notes"""
        breed_notes = {
            'german_shepherd': {
                'alarm_guard': 'German Shepherds are naturally protective and excellent watchdogs',
                'territorial': 'This breed has strong territorial instincts'
            },
            'labrador': {
                'play_invitation': 'Labs are known for their playful, friendly nature',
                'attention_seeking': 'Labradors love human interaction and attention'
            },
            'beagle': {
                'howl_contact': 'Beagles are vocal dogs bred for hunting communication',
                'alarm_guard': 'Beagles will alert you to interesting scents and sounds'
            },
            'chihuahua': {
                'alarm_guard': 'Small dogs like Chihuahuas often have big personalities and strong protective instincts',
                'territorial': 'Despite their size, Chihuahuas can be quite territorial'
            }
        }
        
        return breed_notes.get(breed.lower(), {}).get(intent)
    
    def _get_advice(self, intent: str, tags: List[str]) -> str:
        """Get actionable advice"""
        base_advice = self.actionable_advice.get(intent, [
            "Observe your dog's body language for additional context",
            "Respond to their basic needs and provide appropriate attention"
        ])
        
        # Modify advice based on context tags
        if 'pain_discomfort' in intent and any(tag in ['vet', 'night'] for tag in tags):
            return "Monitor closely and consult your veterinarian if this behavior continues or worsens."
        
        if 'high_energy' in tags:
            return f"{base_advice[0]}. Since they seem energetic, consider physical exercise or mental stimulation."
        
        return base_advice[0] if base_advice else "Monitor the situation and respond appropriately."
    
    def _build_technical_summary(self, request: ExplanationRequest) -> str:
        """Build technical summary for advanced users"""
        
        # Get top secondary predictions
        top_tags = sorted(
            [(tag, prob) for tag, prob in request.tier2_probs.items() if prob > 0.3],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        summary_parts = [
            f"Primary intent: {request.tier1_intent} ({request.tier1_confidence:.1%})"
        ]
        
        if top_tags:
            tag_str = ', '.join([f"{tag} ({prob:.1%})" for tag, prob in top_tags])
            summary_parts.append(f"Context indicators: {tag_str}")
        
        if request.metadata:
            if 'duration' in request.metadata:
                summary_parts.append(f"Duration: {request.metadata['duration']:.1f}s")
            if 'energy' in request.metadata:
                energy_level = 'high' if request.metadata['energy'] > 0.1 else 'medium' if request.metadata['energy'] > 0.05 else 'low'
                summary_parts.append(f"Energy: {energy_level}")
        
        return ' | '.join(summary_parts)
    
    def create_prompt_for_llm(self, request: ExplanationRequest) -> str:
        """Create prompt for external LLM if needed"""
        
        prompt = f"""You are a concise pet behavior explainer. Never make medical claims.

Audio analysis suggests: {request.tier1_intent} (confidence: {request.tier1_confidence:.1%})
Secondary indicators: {', '.join(request.tier2_tags) if request.tier2_tags else 'none'}
Context: Dog breed: {request.breed or 'unknown'}, Environment: {request.environment or 'unknown'}

Generate 1-2 sentences that a 12-year-old can understand explaining what the dog is likely trying to communicate, followed by 2 actionable tips for the owner.

Format:
Explanation: [1-2 sentences about what the dog is communicating]
Tips: 
1. [First actionable tip]
2. [Second actionable tip]"""
        
        return prompt

def test_explainer():
    """Test the explanation system"""
    print("🧪 Testing DogSpeak Explainer")
    
    explainer = DogSpeakExplainer()
    
    # Test case 1: Alarm bark
    request1 = ExplanationRequest(
        tier1_intent='alarm_guard',
        tier1_confidence=0.87,
        tier2_tags=['doorbell', 'stranger', 'indoor'],
        tier2_probs={'doorbell': 0.92, 'stranger': 0.78, 'indoor': 0.65},
        overall_confidence=0.85,
        metadata={'duration': 3.2, 'energy': 0.15},
        breed='german_shepherd',
        environment='indoor'
    )
    
    result1 = explainer.generate_explanation(request1)
    
    print("🔔 Alarm Bark Example:")
    print(f"   Explanation: {result1['explanation']}")
    print(f"   Advice: {result1['advice']}")
    print(f"   Technical: {result1['technical_summary']}")
    
    # Test case 2: Play invitation
    request2 = ExplanationRequest(
        tier1_intent='play_invitation',
        tier1_confidence=0.73,
        tier2_tags=['toy_present', 'high_energy'],
        tier2_probs={'toy_present': 0.81, 'high_energy': 0.89},
        overall_confidence=0.76,
        metadata={'duration': 1.8, 'energy': 0.12},
        breed='labrador'
    )
    
    result2 = explainer.generate_explanation(request2)
    
    print("\n🎾 Play Invitation Example:")
    print(f"   Explanation: {result2['explanation']}")
    print(f"   Advice: {result2['advice']}")
    print(f"   Technical: {result2['technical_summary']}")
    
    # Test LLM prompt generation
    prompt = explainer.create_prompt_for_llm(request1)
    print(f"\n📝 LLM Prompt Example:")
    print(prompt)
    
    return explainer

if __name__ == "__main__":
    test_explainer()
