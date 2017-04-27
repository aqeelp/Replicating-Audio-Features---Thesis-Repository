import numpy as np
import sys
from scipy.cluster.vq import vq, kmeans

if len(sys.argv) != 2:
	sys.exit("Usage: python Reduce_Training_Data.py [number of centroids] [filenames of files to reduce...]")

sample_weights = []
means = [] # array of arrays
sections = [] # array of arrays

for filename in sys.argv[2:]:
	f = open(filename)
	samples = 0
	for line in f:
		vals = line.split()
		if vals[0] == 'SAMPLES':
			sample_weights.append(int(vals[1]))
		elif vals[0] == 'MEAN':
			means.append(map(float, vals[1:]))
		elif vals[0] == 'SECTION':
			float_section = map(float, vals[1:])
			if not np.isnan(float_section).any():
				sections.append(float_section)

# Convert all decibel values to linear amplitudes for proper averaging.
amplitudes = map(lambda x: map((lambda y: (10 ** (y / 20))), x), means)
mean_amp = np.mean(amplitudes)
relative_amplitudes = map(lambda x: map((lambda y: y - mean_amp), x), amplitudes)
relative_amplitudes = np.transpose(relative_amplitudes)

# Perform a weighted mean of the given spectra.
global_weighted_mean = []
for row in relative_amplitudes:
	global_weighted_mean.append(np.average(row, weights=sample_weights))
# Convert back to db:
global_weighted_mean = map((lambda x: 20 * np.log10(x + mean_amp)), global_weighted_mean)

# Output the final average spectrum.
print 'MEAN',
print ' '.join(map(str, global_weighted_mean))

# Use k-means clustering to find the k unique clusters.
sections_as_amplitudes = map(lambda x: map((lambda y: 10 ** (y / 20)), x), sections)
centroids, _ = kmeans(np.asarray(sections_as_amplitudes), int(sys.argv[1]))
for centroid in centroids:
	# Output each cluster.
	print 'CENTROID',
	print ' '.join(map(str, map((lambda x: 20 * np.log10(x)), centroid)))
