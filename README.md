# BulbulNET - Base-unit Analysis

## Two different methods for audio analysis and characterization by data reduction of bird vocal units – White Spectacled Bulbul 

### Introduction –
**Step 1 (analysis)** - These scripts can be used to obtain two different audio analysis – a) “words” analysis - using mel-spectrograms with 35 filters that create one feature vector that describe the entire vocalization. b) “syllables” analysis - extraction of acoustic features, including the fundamental frequency contour which is modeled using Legendre polynomials.

The input are vocal units from "words" / "syllables" directory, which are divided to vocal unit types in advance.
The output is a data frame- word_df / syllable_df which contains the unit's filename, label (unit name by directory), length, and MFB index / acoustic features.

**Step 2 (data reduction)** - applying PCA / tSNE on Bulbul's words / syllables for data reduction.
users can obtain pca, tsne or other methods.

### Dependencies –
- Python 3.8
- librosa
- sklearn


### Launch –
Run each script through the main scripts. 

Examples of use –

Download directories and scripts at the same order as presented here, choose at "main" script which analysis to run by selecting "True":

**Words analysis**

words = True

new_words = False

syllables = False

(in this example only word analysis will run using the files in "words" directory)

**Syllables analysis**

words = False

new_words = False

syllables = True

(In this example only syllable analysis will run using the files from “syllables” directory)

**New words and syllables analysis**

words = False

new_words = True

syllables = True

(In this example both new word analysis and syllable analysis will run using the files from all directories – "random words", "words", "syllables")
