import librosa
import numpy as np
import sys

##############
# Initialize #
##############
samps_per_second = 44100
buff = samps_per_second * 2 # default: buffer = 2 seconds
if len(sys.argv) == 3:
	buff = int(samps_per_second * float(sys.argv[2]))
elif len(sys.argv) != 2:
	sys.exit("Usage: python Extract_Audio_Features.py audio_file.wav [optional: buffer size (in seconds)]")

y, sr = librosa.load(sys.argv[1], sr=samps_per_second)
fundamental = 20
samples_per_minute = 4800

eq_freqs = [20, 40, 60, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 13000, 16500, 20000]

eq_buckets = {}
for f in eq_freqs:
	eq_buckets[f] = []

# Perform an STFT to extract frequency data.
win_size = sr / fundamental
D = np.abs(librosa.stft(y, n_fft=win_size))

# Bucketize all STFT frequencies to their nearest neighbor in set of EQ frequencies (eq_freqs)
for i in range(0, len(D)):
	bucket = 0
	dist = 40000
	freq = i * fundamental
	for f in eq_freqs: # Iterate over EQ freqs. O(1) but still 30ish to search.
		if abs(freq - f) <= dist: # Favor high if we're between
			dist = abs(freq - f)
			bucket = f
	eq_buckets[bucket].append(i)

# Put all the frequency data into bucketized array
# Average STFT frequencies that are all closest to the same EQ freq.
bucketized_data = []
for i, e in enumerate(eq_freqs):
	row = []
	b = eq_buckets[e]
	for t in range(0, len(D[0])):
		av = 0
		for f in b:
			av = av + D[f,t]
		av = av / len(b)
		row.append(av)
	bucketized_data.append(row)

# Extract important data: total number of samples, maximum amplitude, and the average frequency spectrum.
stft_samples = len(bucketized_data[0])
ref_amplitude = np.max(bucketized_data) ** 2
global_mean = librosa.amplitude_to_db(map(np.mean, bucketized_data), ref=ref_amplitude)

# Output number of samples and the mean frequency spectrum.
print 'SAMPLES', stft_samples
print 'MEAN', ' '.join(map(str, global_mean))

# Consider each buffer-sized window of the audio to be a unique section.
# Output the relative offset that this section's frequency spectrum has from the total average spectrum.
stft_buff = stft_samples / (len(y) / (buff / 2))
for start in xrange(0, stft_samples, stft_buff / 2):
	end = start + stft_buff
	if len(bucketized_data[start:]) < stft_buff:
		end = -1

	this_section = []
	for row in bucketized_data:
		row_section = row[start:end]
		this_section.append(np.mean(row_section))

	diff = [x - y for x, y in zip(librosa.amplitude_to_db(this_section, ref=ref_amplitude), global_mean)]
	print 'SECTION', ' '.join(map(str, diff))
