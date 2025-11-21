#A Set of Functions to be used to create a reference distribution

import numpy as np
import masks
import DataLoader
import matplotlib.pyplot as plt
from scipy import stats

# Takes an array of numpy array as volumes, Kernel is a single numpy volume
def kernelSegmentation(volume, kernel):
    filteredVolume = []
    for sample in volume:
        filteredSample = np.where(kernel == 1.0,sample,-1)
        filteredVolume.append(filteredSample)

    return filteredVolume

# averages out the intensity of a masked/filtered volume
# Takes an array of numpy volumes
def averageIntensityNoZero(volume):
    avgArray = []
    for sample in volume:
        avg = np.mean(sample,where=(sample!=-1))
        avgArray.append(avg)
    return avgArray

# Plots data
def plotDistribution(data):
    quant = np.quantile(data,[0.25,0.5,0.75])
    plt.boxplot(data)
    # Add labels and title
    plt.title("Box Plot")
    # Display the plot
    plt.show()

    # 2. Create Histogram (Bar Plot)
    plt.hist(data, bins=10, density=True, color='g', label='Histogram of Data')
    plt.show()

    return quant


def verifyNormality(data):
    stat,pvalue = stats.shapiro(data)
    print(stat)
    print(pvalue)
    if pvalue > 0.05:
        print("Data appears normally distributed (Shapiro-Wilk test).")
    else:
        print("Data does not appear normally distributed (Shapiro-Wilk test).")
    return

# Averages out an array of numpy arrays inot one numpy array
def createReference(data):
    reference = np.mean(data,axis=0)
    return reference

def createsamples(num):
    # Sample Kernel & Volume
    kernel = np.zeros((20,100,100,100))
    kernel[:,0:30,0:30,0:30] = 1
    volume = []
    # 60 fake 4D volumes
    for x in range(num):
        volume.append(np.random.rand(20,100,100,100))

    return kernel,volume


def main():

    # 60 fake 4D volumes
    kernel,volume = createsamples(60)

    # Apply the Mask onto the volume
    filtered = kernelSegmentation(volume,kernel)

    print(filtered[0].shape)

    reference = createReference(filtered)
    print(reference.shape)
    print(reference)

    # Average of the filtered volume
    avg = averageIntensityNoZero(filtered)
    print(avg)

    quant = plotDistribution(avg)
    print(quant)
    
    verifyNormality(avg)

    return


main()
