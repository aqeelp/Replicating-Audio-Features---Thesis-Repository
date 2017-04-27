import librosa
import numpy as np
import sys
import os
import subprocess
from scipy.cluster.vq import vq, kmeans
from scipy.spatial import distance

##############
# Initialize #
##############
samps_per_second = 44100
buff = samps_per_second * 2 # default: buffer = 2 seconds
ref = 'max'
if len(sys.argv) == 5: # Optional buffer size parameter
	buff = int(samps_per_second * float(sys.argv[4]))
elif len(sys.argv) != 4:
	sys.exit("Usage: python replicate_given_frequency.py input_file.wav output_file.wav freq_file [buff size (in seconds)]")

ideal_rms = 0.15 # Ideal root mean square amplitude of the result.
stft_samples_per_minute = 4800 # Approximate samples per minute in the STFT with a fundamental frequency of 20hz.
fundamental = 20 # 20hz is the fundamental frequency of the STFTs
filename = sys.argv[1]
eq_freqs = [20, 40, 60, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 13000, 16500, 20000]

#################################
# Start by analyzing given file #
#################################
y, sr = librosa.load(filename, sr=44100)
freqs_file = sys.argv[3]

# Get all the freq buckets together...
given_freqs = {}
centroids = []
f = open(freqs_file, 'r')
for line in f:
	splits = line.split()
	if splits[0] == 'MEAN':
		for freq, val in zip(eq_freqs, splits[1:]):
			given_freqs[freq] = float(val)
	elif splits[0] == 'CENTROID':
		centroids.append(map(float, splits[1:]))

# Function for finding the nearest centroid given an offset vector.
def nearest_centroid_to(x):
	nearest = (centroids[0], distance.euclidean(tuple(x), tuple(centroids[0])))
	for centroid in centroids[1:]:
		this_dist = distance.euclidean(tuple(x), tuple(centroid))
		if this_dist < nearest[1]:
			nearest = (centroid, this_dist)
	centroid_as_dict = {}
	for freq, val in zip(eq_freqs, nearest[0]):
		centroid_as_dict[freq] = val
	return centroid_as_dict

# Create directory to dump temp info
dir_title = sys.argv[1].split('.')[0].replace(' ','_') + "_splits"
try:
	directory = os.mkdir(dir_title)
except OSError:
	subprocess.call(["rm", "-rf", dir_title])
	directory = os.mkdir(dir_title)

# Split the given audio file into buffer-based chunks.
half_buff = buff / 2
y, sr = librosa.load(sys.argv[1], sr=44100)
total_length = len(y)
for i in xrange(0, len(y), half_buff):
	if i + buff > len(y):
		audio = y[i:]
	else:
		audio = y[i:i + buff]
	librosa.output.write_wav(dir_title + "/file" + str(i / half_buff) + ".wav", audio, sr)

# Bucketize each resulting frequency in the STFT into the nearest EQ frequency (eq_freqs).
eq_buckets = {}
for f in eq_freqs:
	eq_buckets[f] = []
for freq in xrange(0, 22050, fundamental):
	bucket = 0
	dist = 40000
	for f in eq_freqs: # Iterate over EQ freqs. O(1) but still 30ish to search.
		if abs(freq - f) <= dist: # Favor high if we're between
			dist = abs(freq - f)
			bucket = f
	eq_buckets[bucket].append(freq / fundamental)

# Perform an STFT to find the global max amplitude of the audio file.
win_size = sr / fundamental
D = np.abs(librosa.stft(y, n_fft=win_size))
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
ref_amplitude = np.max(bucketized_data) ** 2

global_mean = librosa.amplitude_to_db(map(np.mean, bucketized_data), ref=ref_amplitude)
print global_mean.tolist()

# Now apply EQing to each chunk individually
for i in range(0, total_length / half_buff):
	filename = dir_title + "/file" + str(i) + ".wav"
	this_buff, cur_sr = librosa.load(filename, sr=44100)

	###################################################
	# Open the given file and extract frequency data. #
	###################################################
	win_size = cur_sr / fundamental
	D = np.abs(librosa.stft(this_buff, n_fft=win_size))

	# Bucketize into the relevant frequencies
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

	this_freqs = {}
	for row, freq in zip(bucketized_data, eq_freqs):
		this_freqs[freq] = librosa.amplitude_to_db(np.mean(row), ref=ref_amplitude)

	# Find the offset from the global mean and use the nearest centroid as the proper offset to apply.
	diff_from_mean = []
	for freq, mean_val in zip(eq_freqs, global_mean):
		diff_from_mean.append(this_freqs[freq] - mean_val)
	offset = nearest_centroid_to(diff_from_mean)

	# Compute the differences from the given text file and the audio file.
	max_increase = 0.0
	for freq in eq_freqs:
		diff = given_freqs[freq] - this_freqs[freq] + offset[freq]
		this_freqs[freq] = diff
		if diff > max_increase:
			max_increase = diff

	###################################
	# Apply the EQ and overwrite file #
	###################################
	command = ["sox", filename, filename[:-4] + "_tmp.wav", 'vol', '-' + str(max_increase + 15.0), 'db']

	# Generate the equalizers, i.e. band boosts
	for i in range(1, len(eq_freqs) - 1):
		freq = eq_freqs[i]
		command.append("equalizer")
		command.append(str(freq))
		command.append("3q")
		diff = this_freqs[freq]
		command.append(str(diff))

	# The lowest EQ frequency will be a lower shelf.
	lo = eq_freqs[0]
	command.append("bass")
	command.append(str(this_freqs[lo]))
	command.append(str(lo))
	command.append("3q")

	# The highest EQ frequency will be an upper shelf.
	hi = eq_freqs[-1]
	command.append("treble")
	command.append(str(this_freqs[hi]))
	command.append(str(hi))
	command.append("3q")
	
	# Apply equalization.
	subprocess.call(command)

	# Normalize the audio of the file by first performing a root mean square analysis.
	command = ['sox', filename[:-4] + "_tmp.wav", '-n', 'stat']
	res = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=False)
	rms_amp = float(res.split('\n')[8].split()[2])
	this_volume = 20 * np.log10(rms_amp)
	ideal_volume = 20 * np.log10(ideal_rms)

	# Apply amplitude normalization.
	subprocess.call(['sox', filename[:-4] + '_tmp.wav', filename[:-4] + '_normalized.wav', 'vol', str(ideal_volume - this_volume - 10), 'db'])


################################
# Stitch together final result #
################################
outfile = sys.argv[2]
subprocess.call(['cp', dir_title + '/file0_normalized.wav', outfile])
temp = dir_title + '/temp.wav'
init_fst = float(buff) / samps_per_second
snd = init_fst / 4
inc = init_fst / 2
for i in range(1, total_length / half_buff):
	this_file = dir_title + "/file" + str(i) + "_normalized.wav"
	# Use an equal power crossfade to combine overlapping slices of audio:
	command = ['sox', outfile, this_file, temp, 'splice', '-q', str(init_fst + ((i - 1) * inc)) + ',' + str(snd)]
	subprocess.call(command)
	subprocess.call(['cp', temp, outfile])

############
# Clean up #
############
subprocess.call(['rm', '-rf', dir_title])