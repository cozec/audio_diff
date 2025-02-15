# Audio Diff 

An implementation of audio comparison using mel-spectrograms and the Longest Common Subsequence (LCS) algorithm, demonstrated through three modification cases.

## Algorithm Overview

### Core Components

1. **Feature Extraction**
   - Mel-spectrogram conversion
   - Parameters:
     - n_mels: 128 (frequency bins)
     - hop_length: 512 (frame shift)
     - win_length: 2048 (frame size)
     - Sample rate: 16kHz

2. **Distance Calculation**
   - Frame-to-frame Euclidean distance
   - Distance matrix computation
   - Adaptive thresholding

3. **Sequence Alignment**
   - LCS algorithm implementation
   - Dynamic programming approach
   - Matching path identification

## Demo Cases

### 1. Word Replacement [checkout: demo_replacement.ipynb]
**Purpose:** Demonstrate word substitution while maintaining context

**Implementation:**
- Original: "the big cat"
- Modified: "the small cat"
- Process:
  1. Generate audio segments
  2. Add 200ms silence between words
  3. Compare sequences
  4. Identify replaced segment

**Expected Results:**
- Matching segments at start/end
- Different segment in middle
- Preserved timing structure

!python create_demo_replacement.py

### 2. Word Insertion [checkout: demo_insertion.ipynb]
**Purpose:** Demonstrate word addition between existing words

**Implementation:**
- Original: "the cat"
- Modified: "the big cat"
- Process:
  1. Generate base audio
  2. Insert new word
  3. Add silence buffers
  4. Analyze timeline expansion

**Expected Results:**
- Matching segments before/after insertion
- New segment identified
- Timeline shift detection

!python create_demo_insertion.py

### 3. Word Deletion [checkout: demo_deletion.ipynb]
**Purpose:** Demonstrate word removal from sequence

**Implementation:**
- Original: "the big cat"
- Modified: "the cat"
- Process:
  1. Generate full sequence
  2. Remove middle word
  3. Maintain silence buffers
  4. Analyze timeline compression

**Expected Results:**
- Matching segments preserved
- Deleted segment identified
- Timeline compression detected

!python create_demo_deletion.py 

## Visualization Outputs

### 1. Waveform Connections
**Purpose:** Show temporal alignment
- Dual waveform display
- Matching point connections
- Color-coded segments
  - Green: Matching
  - Red: Modified
  - Gray: Connections

### 2. Distance Matrix
**Purpose:** Display similarity patterns
- Frame-level distances
- Alignment path
- Modification regions

### 3. Alignment Analysis
**Purpose:** Detailed matching visualization
- Temporal correspondence
- Segment boundaries
- Modification highlights

### 4. Distance Histogram
**Purpose:** Threshold analysis
- Distance distribution
- Similarity patterns
- Threshold selection

## Technical Details

### Audio Processing
