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

def create_demo_deletion():
    """Create demo audio files with word deletion"""
    sr = 16000  # Sample rate
    
    try:
        # Generate synthetic speech for words
        y1 = text_to_audio("the", sr=sr)
        y2 = text_to_audio("big", sr=sr)  # This word will be deleted
        y3 = text_to_audio("cat", sr=sr)
        
        # Add small silence between words
        silence = np.zeros(int(0.2 * sr))  # 200ms silence
        
        # Create original audio with three words
        original_audio = np.concatenate([y1, silence, y2, silence, y3])
        
        # Create modified audio by removing the middle word
        modified_audio = np.concatenate([y1, silence, y3])
        
        # Save audio files
        sf.write('original_deletion.wav', original_audio, sr)
        sf.write('modified_deletion.wav', modified_audio, sr)
        
        print("\nDeletion Demo Files Created:")
        print(f"Original audio: 'the big cat'")
        print(f"Modified audio: 'the cat' (middle word deleted)")
        
        return 'original_deletion.wav', 'modified_deletion.wav'
        
    except Exception as e:
        print(f"Error creating deletion demo: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        original_file, modified_file = create_demo_deletion()
        from compare_audio_mel import MelSpectrogramComparator
        comparator = MelSpectrogramComparator(original_file, modified_file)
        comparator.compare()
    except Exception as e:
        print(f"Error: {str(e)}") 