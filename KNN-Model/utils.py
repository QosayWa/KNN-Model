import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def load(file):
    mat = scipy.io.loadmat(file)
    return mat['Xtrain'], mat['Ytrain'], mat['Xvalid'], mat['Yvalid'], mat['Xtest']


def plot_sample(image_vector, digit):
    image_square = image_vector.reshape(28, 28).T  # transpose the image
    plt.imshow(image_square)
    plt.title("Digit: {}".format(*digit))
    plt.show()


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return (distance)**0.5


array1=np.array([1,2,3,88,3,2,5,9999999])

print(array1.argmax())