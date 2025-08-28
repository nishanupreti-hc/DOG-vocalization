import numpy as np

class DogTranslator:
    def __init__(self):
        # Dog vocalization meanings based on context and audio features
        self.translations = {
            'bark': {
                'high_pitch_short': "Alert! Someone's here!",
                'high_pitch_rapid': "I'm excited! Play with me!",
                'low_pitch_slow': "Warning! Stay away!",
                'medium_pitch': "Hey! Pay attention to me!",
                'default': "I'm trying to communicate!"
            },
            'whine': {
                'high_pitch': "I need something! Please help!",
                'rising_tone': "I'm anxious or worried",
                'soft': "I want attention or affection",
                'default': "I'm uncomfortable or need something"
            },
            'growl': {
                'low_rumble': "I'm warning you - back off!",
                'soft_growl': "I'm not sure about this situation",
                'play_growl': "This is fun! Let's play!",
                'default': "I'm feeling defensive or threatened"
            },
            'howl': {
                'long_sustained': "I'm lonely! Where is everyone?",
                'rising_falling': "I'm responding to sounds/sirens",
                'group_howl': "I'm part of the pack!",
                'default': "I'm communicating over long distance"
            },
            'whimper': {
                'soft': "I'm in pain or distressed",
                'repeated': "I'm very anxious or scared",
                'default': "I need comfort and reassurance"
            }
        }
        
        # Emotional context based on audio features
        self.emotions = {
            'excited': "😄 Your dog is happy and energetic!",
            'anxious': "😰 Your dog seems worried or stressed",
            'playful': "🎾 Your dog wants to play!",
            'alert': "👀 Your dog is being watchful",
            'defensive': "🛡️ Your dog feels threatened",
            'lonely': "💔 Your dog misses companionship",
            'content': "😌 Your dog is relaxed and happy"
        }
    
    def analyze_audio_context(self, audio, sr=22050):
        """Analyze audio features to determine context"""
        import librosa
        
        # Extract features for context analysis
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        tempo = librosa.beat.tempo(y=audio, sr=sr)[0] if len(audio) > sr else 60
        energy = np.mean(np.abs(audio))
        duration = len(audio) / sr
        
        # Determine pitch level
        if spectral_centroid > 2500:
            pitch_level = 'high_pitch'
        elif spectral_centroid < 1000:
            pitch_level = 'low_pitch'
        else:
            pitch_level = 'medium_pitch'
        
        # Determine rhythm/speed
        if tempo > 120 or duration < 0.5:
            rhythm = 'rapid' if 'bark' in str(pitch_level) else 'short'
        elif tempo < 60 or duration > 3.0:
            rhythm = 'slow' if 'bark' in str(pitch_level) else 'sustained'
        else:
            rhythm = 'medium'
        
        # Determine energy level
        energy_level = 'high' if energy > 0.1 else 'soft'
        
        return {
            'pitch_level': pitch_level,
            'rhythm': rhythm,
            'energy_level': energy_level,
            'duration': duration,
            'spectral_centroid': spectral_centroid,
            'tempo': tempo
        }
    
    def get_emotion(self, vocalization, context):
        """Determine emotional state"""
        if vocalization == 'bark':
            if context['pitch_level'] == 'high_pitch' and context['rhythm'] == 'rapid':
                return 'excited'
            elif context['energy_level'] == 'high':
                return 'alert'
            else:
                return 'playful'
        
        elif vocalization == 'whine':
            return 'anxious'
        
        elif vocalization == 'growl':
            if context['energy_level'] == 'soft':
                return 'playful'
            else:
                return 'defensive'
        
        elif vocalization == 'howl':
            return 'lonely'
        
        return 'content'
    
    def translate(self, vocalization, audio=None, sr=22050):
        """Translate dog vocalization to English"""
        
        if audio is not None:
            # Analyze audio context
            context = self.analyze_audio_context(audio, sr)
            
            # Get specific translation based on context
            voc_translations = self.translations.get(vocalization, {})
            
            # Try to find specific context match
            translation = None
            if context['pitch_level'] == 'high_pitch' and context['rhythm'] in ['rapid', 'short']:
                translation = voc_translations.get(f"{context['pitch_level']}_{context['rhythm']}")
            elif context['pitch_level'] == 'low_pitch':
                translation = voc_translations.get(f"{context['pitch_level']}_{context['rhythm']}")
            elif context['energy_level'] == 'soft':
                translation = voc_translations.get('soft')
            
            # Fallback to default
            if not translation:
                translation = voc_translations.get('default', "I'm trying to communicate!")
            
            # Get emotion
            emotion = self.get_emotion(vocalization, context)
            emotion_text = self.emotions.get(emotion, "")
            
            return {
                'translation': translation,
                'emotion': emotion_text,
                'context': {
                    'pitch': context['pitch_level'].replace('_', ' '),
                    'duration': f"{context['duration']:.1f}s",
                    'energy': context['energy_level']
                }
            }
        
        else:
            # Basic translation without audio analysis
            return {
                'translation': self.translations.get(vocalization, {}).get('default', "I'm communicating!"),
                'emotion': "😊 Your dog is expressing themselves",
                'context': {'note': 'Audio analysis needed for detailed context'}
            }
    
    def get_behavioral_advice(self, vocalization, emotion):
        """Provide behavioral advice based on translation"""
        advice = {
            'bark': {
                'excited': "Engage in play or exercise to channel their energy positively!",
                'alert': "Check what caught their attention - they're doing their job as a watchdog!",
                'playful': "Great time for interactive games or training sessions!"
            },
            'whine': {
                'anxious': "Provide comfort and check if they need something (food, water, bathroom).",
                'default': "Look for what they might need - attention, comfort, or basic needs."
            },
            'growl': {
                'defensive': "Give them space and identify what's making them uncomfortable.",
                'playful': "Join in the fun! They're inviting you to play."
            },
            'howl': {
                'lonely': "Spend quality time together or consider if they need more social interaction."
            }
        }
        
        return advice.get(vocalization, {}).get(emotion, "Observe your dog's body language for additional context.")
