import numpy as np
import librosa
import torch
import random
from scipy import signal
from typing import List, Tuple

class AdvancedAudioAugmentation:
    """Advanced augmentation techniques for higher accuracy"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        
        # Load room impulse responses for reverb
        self.reverb_irs = self._generate_reverb_irs()
        
        # Background noise samples
        self.noise_samples = self._generate_noise_samples()
    
    def _generate_reverb_irs(self):
        """Generate synthetic room impulse responses"""
        irs = []
        
        # Small room
        t = np.linspace(0, 0.5, int(0.5 * self.sr))
        ir_small = np.exp(-t * 8) * np.random.randn(len(t)) * 0.1
        irs.append(ir_small)
        
        # Large room
        t = np.linspace(0, 1.2, int(1.2 * self.sr))
        ir_large = np.exp(-t * 3) * np.random.randn(len(t)) * 0.05
        irs.append(ir_large)
        
        return irs
    
    def _generate_noise_samples(self):
        """Generate background noise samples"""
        noises = []
        
        # Traffic noise (low-frequency rumble)
        t = np.linspace(0, 5, 5 * self.sr)
        traffic = np.sum([np.sin(2 * np.pi * f * t) * np.exp(-f/100) 
                         for f in range(20, 200, 10)], axis=0)
        traffic += np.random.randn(len(traffic)) * 0.1
        noises.append(traffic * 0.05)
        
        # TV/Radio chatter (mid-frequency)
        chatter = np.random.randn(3 * self.sr) * 0.02
        chatter = signal.butter(4, [300, 3000], btype='band', fs=self.sr, output='sos')
        chatter = signal.sosfilt(chatter, np.random.randn(3 * self.sr))
        noises.append(chatter * 0.03)
        
        return noises
    
    def mixup_augmentation(self, audio1, audio2, alpha=0.2):
        """Mixup augmentation between two audio samples"""
        lam = np.random.beta(alpha, alpha)
        
        # Ensure same length
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        mixed_audio = lam * audio1 + (1 - lam) * audio2
        return mixed_audio, lam
    
    def specaugment(self, spectrogram, freq_mask_param=15, time_mask_param=25):
        """SpecAugment for spectrograms"""
        spec = spectrogram.copy()
        
        # Frequency masking
        freq_mask_size = random.randint(0, freq_mask_param)
        freq_start = random.randint(0, spec.shape[0] - freq_mask_size)
        spec[freq_start:freq_start + freq_mask_size, :] = 0
        
        # Time masking
        time_mask_size = random.randint(0, time_mask_param)
        time_start = random.randint(0, spec.shape[1] - time_mask_size)
        spec[:, time_start:time_start + time_mask_size] = 0
        
        return spec
    
    def add_reverb(self, audio):
        """Add realistic reverb to audio"""
        ir = random.choice(self.reverb_irs)
        
        # Convolve with impulse response
        reverb_audio = signal.convolve(audio, ir, mode='same')
        
        # Mix with original
        mix_ratio = random.uniform(0.1, 0.4)
        return (1 - mix_ratio) * audio + mix_ratio * reverb_audio
    
    def add_background_noise(self, audio, snr_db_range=(10, 25)):
        """Add realistic background noise"""
        noise = random.choice(self.noise_samples)
        
        # Random segment of noise
        if len(noise) > len(audio):
            start_idx = random.randint(0, len(noise) - len(audio))
            noise_segment = noise[start_idx:start_idx + len(audio)]
        else:
            # Repeat noise if too short
            repeats = len(audio) // len(noise) + 1
            noise_segment = np.tile(noise, repeats)[:len(audio)]
        
        # Set SNR
        target_snr_db = random.uniform(*snr_db_range)
        
        # Calculate power
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise_segment ** 2)
        
        # Scale noise to achieve target SNR
        snr_linear = 10 ** (target_snr_db / 10)
        noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
        
        return audio + noise_scale * noise_segment
    
    def vocal_tract_length_perturbation(self, audio, alpha_range=(0.9, 1.1)):
        """Simulate different dog sizes via VTLP"""
        alpha = random.uniform(*alpha_range)
        
        # Warp frequency axis
        stft = librosa.stft(audio, n_fft=1024, hop_length=256)
        
        # Frequency warping (simplified)
        warped_stft = np.zeros_like(stft)
        freqs = np.linspace(0, 1, stft.shape[0])
        
        for i, freq in enumerate(freqs):
            warped_freq = freq ** alpha
            warped_idx = int(warped_freq * (stft.shape[0] - 1))
            if warped_idx < stft.shape[0]:
                warped_stft[i] = stft[warped_idx]
        
        return librosa.istft(warped_stft, hop_length=256)
    
    def dynamic_range_compression(self, audio, ratio=4.0, threshold=0.1):
        """Apply dynamic range compression"""
        # Simple compressor
        compressed = np.copy(audio)
        
        # Find samples above threshold
        above_threshold = np.abs(audio) > threshold
        
        # Apply compression
        compressed[above_threshold] = np.sign(audio[above_threshold]) * (
            threshold + (np.abs(audio[above_threshold]) - threshold) / ratio
        )
        
        return compressed
    
    def formant_shifting(self, audio, shift_factor_range=(0.8, 1.2)):
        """Shift formant frequencies"""
        shift_factor = random.uniform(*shift_factor_range)
        
        # Use librosa's pitch shift as approximation
        # (Real formant shifting would require more complex processing)
        n_steps = 12 * np.log2(shift_factor)
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def multi_band_compression(self, audio):
        """Apply multi-band compression"""
        # Split into frequency bands
        low_cutoff = 800
        high_cutoff = 2400
        
        # Design filters
        sos_low = signal.butter(4, low_cutoff, btype='low', fs=self.sr, output='sos')
        sos_mid = signal.butter(4, [low_cutoff, high_cutoff], btype='band', fs=self.sr, output='sos')
        sos_high = signal.butter(4, high_cutoff, btype='high', fs=self.sr, output='sos')
        
        # Filter into bands
        low_band = signal.sosfilt(sos_low, audio)
        mid_band = signal.sosfilt(sos_mid, audio)
        high_band = signal.sosfilt(sos_high, audio)
        
        # Apply different compression to each band
        low_compressed = self.dynamic_range_compression(low_band, ratio=2.0)
        mid_compressed = self.dynamic_range_compression(mid_band, ratio=4.0)
        high_compressed = self.dynamic_range_compression(high_band, ratio=6.0)
        
        # Recombine
        return low_compressed + mid_compressed + high_compressed
    
    def augment_batch(self, audio_batch, labels_batch, augmentation_factor=3):
        """Apply comprehensive augmentation to a batch"""
        augmented_audio = []
        augmented_labels = []
        
        for audio, label in zip(audio_batch, labels_batch):
            # Original sample
            augmented_audio.append(audio)
            augmented_labels.append(label)
            
            # Generate augmentations
            for _ in range(augmentation_factor):
                aug_audio = audio.copy()
                
                # Random combination of augmentations
                augmentations = [
                    ('pitch', 0.3),
                    ('time', 0.3),
                    ('reverb', 0.2),
                    ('noise', 0.4),
                    ('vtlp', 0.2),
                    ('compression', 0.3),
                    ('formant', 0.2)
                ]
                
                for aug_name, prob in augmentations:
                    if random.random() < prob:
                        if aug_name == 'pitch':
                            n_steps = random.uniform(-2, 2)
                            aug_audio = librosa.effects.pitch_shift(aug_audio, sr=self.sr, n_steps=n_steps)
                        elif aug_name == 'time':
                            rate = random.uniform(0.85, 1.15)
                            aug_audio = librosa.effects.time_stretch(aug_audio, rate=rate)
                        elif aug_name == 'reverb':
                            aug_audio = self.add_reverb(aug_audio)
                        elif aug_name == 'noise':
                            aug_audio = self.add_background_noise(aug_audio)
                        elif aug_name == 'vtlp':
                            aug_audio = self.vocal_tract_length_perturbation(aug_audio)
                        elif aug_name == 'compression':
                            aug_audio = self.multi_band_compression(aug_audio)
                        elif aug_name == 'formant':
                            aug_audio = self.formant_shifting(aug_audio)
                
                augmented_audio.append(aug_audio)
                augmented_labels.append(label)
        
        return augmented_audio, augmented_labels

class SmartAugmentationSelector:
    """Intelligently select augmentations based on class imbalance"""
    
    def __init__(self):
        self.class_counts = {}
        self.augmentation_strategies = {}
    
    def analyze_dataset(self, labels):
        """Analyze dataset for class imbalance"""
        unique, counts = np.unique(labels, return_counts=True)
        self.class_counts = dict(zip(unique, counts))
        
        # Determine augmentation needs
        max_count = max(counts)
        for class_name, count in self.class_counts.items():
            augmentation_factor = max_count // count
            self.augmentation_strategies[class_name] = min(augmentation_factor, 5)  # Cap at 5x
    
    def get_augmentation_factor(self, label):
        """Get augmentation factor for specific class"""
        return self.augmentation_strategies.get(label, 1)

def test_advanced_augmentation():
    """Test advanced augmentation system"""
    print("🎵 Testing Advanced Audio Augmentation")
    print("=" * 40)
    
    # Create augmenter
    augmenter = AdvancedAudioAugmentation()
    
    # Generate test audio
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Synthetic dog bark
    bark = (
        0.5 * np.sin(2 * np.pi * 800 * t) +
        0.3 * np.sin(2 * np.pi * 1600 * t) +
        0.2 * np.random.randn(len(t)) * 0.1
    )
    
    # Test different augmentations
    print("✅ Original audio shape:", bark.shape)
    
    # Reverb
    reverb_bark = augmenter.add_reverb(bark)
    print("✅ Reverb applied")
    
    # Background noise
    noisy_bark = augmenter.add_background_noise(bark)
    print("✅ Background noise added")
    
    # VTLP
    vtlp_bark = augmenter.vocal_tract_length_perturbation(bark)
    print("✅ VTLP applied")
    
    # Multi-band compression
    compressed_bark = augmenter.multi_band_compression(bark)
    print("✅ Multi-band compression applied")
    
    # Batch augmentation
    batch_audio = [bark] * 3
    batch_labels = ['bark'] * 3
    
    aug_audio, aug_labels = augmenter.augment_batch(batch_audio, batch_labels)
    print(f"✅ Batch augmentation: {len(batch_audio)} → {len(aug_audio)} samples")
    
    # Smart augmentation selector
    selector = SmartAugmentationSelector()
    test_labels = ['bark'] * 100 + ['whine'] * 20 + ['growl'] * 50
    selector.analyze_dataset(test_labels)
    
    print("✅ Smart augmentation factors:")
    for label, factor in selector.augmentation_strategies.items():
        print(f"   {label}: {factor}x augmentation")
    
    return augmenter

if __name__ == "__main__":
    test_advanced_augmentation()
