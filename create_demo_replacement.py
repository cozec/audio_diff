import numpy as np
import soundfile as sf
import gtts
import io
import librosa

def text_to_audio(text, sr=16000):
    """Convert text to audio using Google TTS"""
    try:
        tts = gtts.gTTS(text=text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        y, _ = librosa.load(fp, sr=sr)
        return y
    except Exception as e:
        print(f"Error in text_to_audio: {str(e)}")
        raise

def create_demo_replacement():
    """Create demo audio files with word replacement"""
    sr = 16000  # Sample rate
    
    try:
        # Generate synthetic speech for words
        y1 = text_to_audio("the", sr=sr)
        y2_original = text_to_audio("big", sr=sr)
        y2_modified = text_to_audio("small", sr=sr)
        y3 = text_to_audio("cat", sr=sr)
        
        # Add small silence between words
        silence = np.zeros(int(0.2 * sr))  # 200ms silence
        
        # Create original audio with three words
        original_audio = np.concatenate([y1, silence, y2_original, silence, y3])
        
        # Create modified audio by replacing the middle word
        modified_audio = np.concatenate([y1, silence, y2_modified, silence, y3])
        
        # Save audio files
        sf.write('original_replacement.wav', original_audio, sr)
        sf.write('modified_replacement.wav', modified_audio, sr)
        
        print("\nReplacement Demo Files Created:")
        print(f"Original audio: 'the big cat'")
        print(f"Modified audio: 'the small cat' (middle word replaced)")
        
        return 'original_replacement.wav', 'modified_replacement.wav'
        
    except Exception as e:
        print(f"Error creating replacement demo: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        original_file, modified_file = create_demo_replacement()
        from compare_audio_mel import MelSpectrogramComparator
        comparator = MelSpectrogramComparator(original_file, modified_file)
        comparator.compare()
    except Exception as e:
        print(f"Error: {str(e)}") 