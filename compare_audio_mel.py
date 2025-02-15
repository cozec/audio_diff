import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    segment_type: str  # 'common', 'unique_1', or 'unique_2'

class MelSpectrogramComparator:
    def __init__(self, file1_path, file2_path):
        self.file1_path = Path(file1_path)
        self.file2_path = Path(file2_path)
        
        # Mel-spectrogram parameters
        self.sr = 16000
        self.n_mels = 128
        self.n_fft = 2048
        self.win_length = 400
        self.hop_length = 160 # 512
        self.similarity_threshold = 1.0 # 1.5 
        #self.distance_threshold = 0.3
        # Store audio data
        self.sr1 = None
        self.sr2 = None
        self.segments = []

    def load_and_convert_to_mel(self, file_path):
        """Load audio file and convert to mel-spectrogram"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sr)
            
            # Store sample rate
            if file_path == self.file1_path:
                self.sr1 = sr
            else:
                self.sr2 = sr
            
            # Convert to mel-spectrogram
            mel_spect = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=self.n_fft, 
                hop_length=self.hop_length, n_mels=self.n_mels,
                win_length=self.win_length
            )
            
            # Convert to dB scale
            mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
            
            # Normalize
            mel_spect_normalized = (mel_spect_db - mel_spect_db.min()) / (
                mel_spect_db.max() - mel_spect_db.min())
            
            return mel_spect_normalized
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def find_lcs_with_backtrack(self, mel1, mel2):
        """Find LCS and track the path for backtracking using Euclidean distance"""
        m, n = mel1.shape[1], mel2.shape[1]
        dp = np.zeros((m + 1, n + 1))
        # Directions: 0 = none, 1 = diagonal, 2 = up, 3 = left
        directions = np.zeros((m + 1, n + 1), dtype=int)
        
        # Pre-compute distance matrix
        distance_matrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                # Calculate Euclidean distance between mel-spectrogram frames
                distance_matrix[i, j] = np.linalg.norm(mel1[:, i] - mel2[:, j])

        # Adjust threshold for direct Euclidean distance
        # distance_threshold = 0.3  # Lower values indicate more similarity

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                distance = distance_matrix[i-1, j-1]
                
                if distance <= self.similarity_threshold:
                    dp[i, j] = dp[i-1, j-1] + 1
                    directions[i, j] = 1  # diagonal
                else:
                    if dp[i-1, j] >= dp[i, j-1]:
                        dp[i, j] = dp[i-1, j]
                        directions[i, j] = 2  # up
                    else:
                        dp[i, j] = dp[i, j-1]
                        directions[i, j] = 3  # left

        return dp, directions, distance_matrix

    def backtrack_segments(self, directions, mel1, mel2):
        """Backtrack through the directions matrix to identify segments"""
        i, j = directions.shape[0] - 1, directions.shape[1] - 1
        current_segment = None
        segments = []
        map_time_org_mod = []
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and directions[i, j] == 1:  # diagonal (match)
                time = (i - 1) * self.hop_length / self.sr1
                time_j = (j - 1) * self.hop_length / self.sr2
                map_time_org_mod.append((time, time_j))
                if current_segment is None or current_segment.segment_type != 'common':
                    if current_segment is not None:
                        segments.append(current_segment)
                    current_segment = AudioSegment(time, time, 'common')
                current_segment.start_time = time
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or directions[i, j] == 2):  # up (unique in first)
                time = (i - 1) * self.hop_length / self.sr1
                if current_segment is None or current_segment.segment_type != 'unique_1':
                    if current_segment is not None:
                        segments.append(current_segment)
                    current_segment = AudioSegment(time, time, 'unique_1')
                current_segment.start_time = time
                i -= 1
            else:  # left (unique in second)
                time = (j - 1) * self.hop_length / self.sr2
                if current_segment is None or current_segment.segment_type != 'unique_2':
                    if current_segment is not None:
                        segments.append(current_segment)
                    current_segment = AudioSegment(time, time, 'unique_2')
                current_segment.start_time = time
                j -= 1
        
        if current_segment is not None and current_segment.end_time - current_segment.start_time > 0.01:
            segments.append(current_segment)
        
        self.segments = list(reversed(segments))
        self.map_time_org_mod = list(reversed(map_time_org_mod))
        return self.map_time_org_mod

    def plot_alignment_analysis(self, mel1, mel2, directions, distance_matrix, output_path='alignment_analysis.png'):
        """Plot comprehensive alignment analysis in one figure"""
        fig = plt.figure(figsize=(15, 15))
        gs = plt.GridSpec(3, 3, figure=fig)

        # Load audio files
        y1, _ = librosa.load(self.file1_path, sr=self.sr1)
        y2, _ = librosa.load(self.file2_path, sr=self.sr2)
        
        # Left side: Original audio waveform (vertical)
        ax_wave1 = fig.add_subplot(gs[1:, 0])
        ax_wave1.plot(y1, np.arange(len(y1)) / self.sr1, color='gray', alpha=0.7)
        ax_wave1.set_xlabel('Amplitude')
        ax_wave1.set_ylabel('Time (s)')
        ax_wave1.invert_xaxis()
        
        # Top: Modified audio waveform (horizontal)
        ax_wave2 = fig.add_subplot(gs[0, 1:])
        ax_wave2.plot(np.arange(len(y2)) / self.sr2, y2, color='gray', alpha=0.7)
        ax_wave2.set_ylabel('Amplitude')
        ax_wave2.set_xlabel('Time (s)')
        
        # Plot similarity matrix with alignment path
        ax_sim = fig.add_subplot(gs[1:, 1:])
        im = ax_sim.imshow(distance_matrix.T, aspect='auto', origin='lower', 
                          extent=[0, len(y1)/self.sr1, 0, len(y2)/self.sr2],
                          cmap='viridis')
        plt.colorbar(im, ax=ax_sim, label='Similarity Score')
        
        # Collect and plot alignment path
        path_points = []
        i, j = directions.shape[0] - 1, directions.shape[1] - 1
        while i > 0 or j > 0:
            path_points.append((
                i * len(y1) / (directions.shape[0] * self.sr1),
                j * len(y2) / (directions.shape[1] * self.sr2)
            ))
            if i > 0 and j > 0 and directions[i, j] == 1:  # diagonal
                i -= 1
                j -= 1
            elif i > 0 and directions[i, j] == 2:  # up
                i -= 1
            else:  # left
                j -= 1
        path_points.append((0, 0))
        
        path_x, path_y = zip(*reversed(path_points))
        ax_sim.plot(path_x, path_y, 'r-', linewidth=2, label='Alignment Path')
        
        # Mark segments
        for segment in self.segments:
            if segment.segment_type == 'common':
                rect = plt.Rectangle(
                    (segment.start_time, segment.start_time),
                    segment.end_time - segment.start_time,
                    segment.end_time - segment.start_time,
                    fill=False,
                    color='lime',
                    linewidth=2,
                    label='Common Segment' if segment == self.segments[0] else ""
                )
                ax_sim.add_patch(rect)
                
                ax_wave1.axhspan(segment.start_time, segment.end_time, 
                               color='green', alpha=0.3)
                ax_wave2.axvspan(segment.start_time, segment.end_time, 
                               color='green', alpha=0.3)
            else:
                if segment.segment_type == 'unique_1':
                    ax_wave1.axhspan(segment.start_time, segment.end_time, 
                                   color='red', alpha=0.3)
                elif segment.segment_type == 'unique_2':
                    ax_wave2.axvspan(segment.start_time, segment.end_time, 
                                   color='red', alpha=0.3)
        
        ax_sim.set_xlabel('Time - Original Audio (s)')
        ax_sim.set_ylabel('Time - Modified Audio (s)')
        fig.suptitle('Audio Alignment Analysis', fontsize=16)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.3, label='Common Segments'),
            Patch(facecolor='red', alpha=0.3, label='Unique Segments'),
            plt.Line2D([0], [0], color='r', label='Alignment Path')
        ]
        ax_sim.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_distance_matrix(self, mel1, mel2, distance_matrix, output_path='distance_matrix.png'):
        """Plot distance matrix and its distribution"""
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])

        # Plot distance matrix
        ax_matrix = fig.add_subplot(gs[1, 0])
        im = ax_matrix.imshow(distance_matrix.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax_matrix, label='Similarity Score')
        
        # Add labels
        ax_matrix.set_xlabel('Time - Original Audio (s)')
        ax_matrix.set_ylabel('Time - Modified Audio (s)')
        
        # Convert frame indices to time
        n_frames1 = distance_matrix.shape[0]
        n_frames2 = distance_matrix.shape[1]
        
        time_ticks1 = np.linspace(0, n_frames1 * self.hop_length / self.sr1, 5)
        time_ticks2 = np.linspace(0, n_frames2 * self.hop_length / self.sr2, 5)
        
        frame_ticks1 = np.linspace(0, n_frames1, 5)
        frame_ticks2 = np.linspace(0, n_frames2, 5)
        
        ax_matrix.set_xticks(frame_ticks1)
        ax_matrix.set_yticks(frame_ticks2)
        ax_matrix.set_xticklabels([f'{t:.1f}' for t in time_ticks1])
        ax_matrix.set_yticklabels([f'{t:.1f}' for t in time_ticks2])

        # Plot horizontal distribution (for original audio)
        ax_hist1 = fig.add_subplot(gs[0, 0])
        ax_hist1.hist(distance_matrix.mean(axis=1), bins=50, color='blue', alpha=0.6)
        ax_hist1.set_ylabel('Count')
        ax_hist1.set_title('Average Similarity Distribution (Original)')
        
        # Plot vertical distribution (for modified audio)
        ax_hist2 = fig.add_subplot(gs[1, 1])
        ax_hist2.hist(distance_matrix.mean(axis=0), bins=50, orientation='horizontal', color='blue', alpha=0.6)
        ax_hist2.set_xlabel('Count')
        ax_hist2.set_title('Average Similarity Distribution (Modified)')
        
        # Add threshold line to distributions
        ax_hist1.axvline(self.similarity_threshold, color='red', linestyle='--', alpha=0.5, 
                         label=f'Threshold ({self.similarity_threshold})')
        ax_hist2.axhline(self.similarity_threshold, color='red', linestyle='--', alpha=0.5, 
                         label=f'Threshold ({self.similarity_threshold})')
        
        # Add legends
        ax_hist1.legend()
        ax_hist2.legend()
        
        # Add overall title
        plt.suptitle('Distance Matrix Analysis', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_distance_histogram(self, distance_matrix, output_path='distance_histogram.png'):
        """Plot histogram of all distance values"""
        plt.figure(figsize=(10, 6))
        
        # Flatten the distance matrix and plot histogram
        distances = distance_matrix.flatten()
        plt.hist(distances, bins=50, color='blue', alpha=0.7)
        
        # Add vertical line for threshold
        plt.axvline(x=self.similarity_threshold, color='red', linestyle='--', 
                    label=f'Threshold ({self.similarity_threshold:.3f})')
        
        # Add labels and title
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Count')
        plt.title('Distribution of Frame-to-Frame Distances')
        plt.legend()
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_waveform_connections(self, output_path='waveform_connections.png'):
        """Plot waveforms with thick dotted connection lines between matching points in one panel"""
        # Load audio files
        y1, _ = librosa.load(self.file1_path, sr=self.sr1)
        y2, _ = librosa.load(self.file2_path, sr=self.sr2)
        
        # Create figure
        plt.figure(figsize=(15, 6))
        
        # Plot both waveforms in one panel
        time1 = np.arange(len(y1)) / self.sr1
        time2 = np.arange(len(y2)) / self.sr2
        
        # Offset the waveforms vertically for better visualization
        offset = 2
        plt.plot(time1, y1 + offset, color='blue', alpha=0.7, label='Original')
        plt.plot(time2, y2 - offset, color='red', alpha=0.7, label='Modified')
        
        # Draw connection lines between matching points
        if self.map_time_org_mod:
            for time_org, time_mod in self.map_time_org_mod:
                plt.plot([time_org, time_mod], 
                        [y1[int(time_org * self.sr1)] + offset, 
                         y2[int(time_mod * self.sr2)] - offset],
                        color='green', alpha=0.3, linewidth=1.5,
                        linestyle='-', zorder=1)
        
        # Add labels and title
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Audio Waveform Comparison with Matching Points')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adjust y-axis limits to show both waveforms clearly
        plt.ylim(-offset*1.5, offset*1.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def compare(self):
        """Perform comparison and segment identification"""
        print("Converting audio files to mel-spectrograms...")
        mel1 = self.load_and_convert_to_mel(self.file1_path)
        mel2 = self.load_and_convert_to_mel(self.file2_path)

        if mel1 is None or mel2 is None:
            return

        print("Calculating similarity and identifying segments...")
        dp, directions, distance_matrix = self.find_lcs_with_backtrack(mel1, mel2)
        self.backtrack_segments(directions, mel1, mel2)

        # Generate visualizations
        print("Generating visualizations...")
        self.plot_distance_matrix(mel1, mel2, distance_matrix,
                                'distance_matrix.png')
        self.plot_distance_histogram(distance_matrix,
                                   'distance_histogram.png')
        self.plot_waveform_connections('waveform_connections.png')

        # Print results
        print("\nSegment Analysis Results:")
        print("-" * 50)
        print(f"File 1: {self.file1_path}")
        print(f"File 2: {self.file2_path}")
        
        print("\nIdentified Segments:")
        for i, segment in enumerate(self.segments, 1):
            print(f"\nSegment {i}:")
            print(f"Type: {segment.segment_type}")
            print(f"Start time: {segment.start_time:.2f}s")
            print(f"End time: {segment.end_time:.2f}s")
            print(f"Duration: {segment.end_time - segment.start_time:.2f}s")

        print("\nVisualizations saved as:")
        print("- 'distance_matrix.png' (similarity matrix and distributions)")
        print("- 'distance_histogram.png' (distribution of all distances)")
        print("- 'waveform_connections.png' (waveforms with matching points)")

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_audio_mel.py <audio_file1> <audio_file2>")
        sys.exit(1)

    comparator = MelSpectrogramComparator(sys.argv[1], sys.argv[2])
    comparator.compare()

if __name__ == "__main__":
    main()