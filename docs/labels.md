# DogSpeak Translator - Intent Taxonomy

## Tier-1 Intents (Base Classes)

| Intent | Description | Typical Sounds | Context |
|--------|-------------|----------------|---------|
| `alarm_guard` | Alert/protective behavior | Sharp, repetitive barks | Strangers, unusual sounds |
| `territorial` | Defending space/resources | Deep, sustained barks | Other dogs, boundaries |
| `play_invitation` | Requesting play/interaction | High-pitched, bouncy barks | Toys present, owner attention |
| `distress_separation` | Anxiety/loneliness | Whining, howling | Owner leaving, isolation |
| `pain_discomfort` | Physical distress | Whimpers, yelps | Injury, illness |
| `attention_seeking` | Wanting owner focus | Mixed barks/whines | Food time, walk time |
| `whine_appeal` | Polite request | Soft whining | Needs bathroom, wants something |
| `growl_threat` | Warning/aggression | Low rumbling | Feeling threatened |
| `growl_play` | Playful warning | Higher-pitched growl | During play, tug-of-war |
| `howl_contact` | Long-distance communication | Sustained howling | Sirens, other dogs |
| `yip_puppy` | Juvenile vocalizations | High-pitched yelps | Young dogs, excitement |
| `other_unknown` | Unclassified sounds | Various | Unclear context |

## Tier-2 Context Tags

### Environmental Triggers
- `doorbell` - Doorbell or knocking
- `stranger` - Unknown person present
- `owner_arrives` - Owner returning home
- `walk_time` - Pre-walk excitement
- `food_time` - Meal preparation/feeding
- `toy_present` - Toys or play objects visible
- `vet` - Veterinary environment
- `crate` - Crate/kennel related
- `night` - Nighttime/dark conditions
- `other_dog` - Other dogs present
- `thunder` - Thunder/storms
- `fireworks` - Loud sudden noises

### Situational Modifiers
- `indoor` - Inside environment
- `outdoor` - Outside environment
- `high_energy` - Excited state
- `calm` - Relaxed state
- `multiple_dogs` - Pack behavior
- `alone` - Dog is isolated

## Metadata Schema

```yaml
sample_id: str
timestamp: datetime
duration_ms: int
breed: str  # "mixed", "labrador", "german_shepherd", etc.
size: str   # "small", "medium", "large"
age_months: int
environment: str  # "indoor", "outdoor", "car", etc.
distance_to_mic: str  # "close", "medium", "far"
snr_db: float
background_noise: str  # "quiet", "tv", "traffic", "other_dogs"
overlapping_speakers: bool
annotator_confidence: float  # 0.0-1.0
tier1_intent: str
tier2_tags: List[str]
free_text_notes: str
```

## Quality Control Guidelines

### Annotation Standards
- **High Confidence (0.8-1.0)**: Clear intent, good audio quality, obvious context
- **Medium Confidence (0.5-0.8)**: Likely intent, some ambiguity or noise
- **Low Confidence (0.0-0.5)**: Unclear intent, poor quality, multiple possibilities

### Inter-Annotator Agreement
- Minimum 2 annotators per sample
- Disagreement threshold: >0.3 difference in confidence
- Escalation: 3rd annotator for disputed samples
- Target Kappa: >0.7 for tier-1 intents

### Exclusion Criteria
- Audio <1s or >15s duration
- SNR <-5dB (unless specifically collecting noisy samples)
- Non-dog vocalizations (human speech, other animals)
- Corrupted or incomplete recordings
