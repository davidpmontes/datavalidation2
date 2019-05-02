import os
from pandas import DataFrame
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
import statistics
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

NUM_IMAGES_PER_GROUP = 2

class DataValidator():
    # given image number, contains list of x coordinates for every box from every group
    x = []

    # given image number, contains list of y coordinates for every box from every group
    y = []


    kaverage = []
    knum = []

    # given image number, given index for box, return centroid coordinate
    centroids = []
    labels = []

    # given image number, dictionary for centroid's index to box label popular
    listOfDictionariesForCentroidsToBoxLabelsPopular = []
    listOfDictionariesForCentroidsToBoxLabels = []

    # given a coordinate, => group name and box name
    coordinatesToGroupNameAndBoxName = {}

    # given a coordinate, => group name and image name
    coordinatesToGroupNameAndImage = {}

    # given image number, dictionary for group name to distance
    GroupNamesToDistanceList = []

    def initializeArrays(self):

        for i in range(0, NUM_IMAGES_PER_GROUP):
            self.x.append([])
            self.y.append([])
            self.kaverage.append([])

    def readFiles(self):
        testing_directory_path = os.path.expanduser("~/Desktop/TestThis4")
    
        group_folders = os.listdir(testing_directory_path)

        for group_folder in group_folders:
            group_path = os.path.join(testing_directory_path, group_folder)

            numimage = 0

            if (os.path.isdir(group_path)):
                images = os.listdir(group_path)
                
                for image in images:
                    image_path = os.path.join(group_path, image)
                    extension = os.path.splitext(image)[1]

                    if (extension == ".xml"):
                        numboxes = 0
                        filename = ""

                        tree = ET.parse(image_path)
                        root = tree.getroot()

                        for child in root:
                            if child.tag == "filename":
                                filename = child.text
                            if child.tag == "object":
                                boxName = ""
                                numboxes += 1
                                for childp in child:
                                    if childp.tag == "name":
                                        boxName = childp.text

                                    if childp.tag == "polygon":
                                        xtotal = 0
                                        ytotal = 0

                                        for childpp in childp:
                                            if childpp.tag == "pt":
                                                xtotal += int(childpp[0].text)
                                                ytotal += int(childpp[1].text)
                                        xcenter = xtotal / 4
                                        ycenter = ytotal / 4
                                        self.x[numimage].append(xcenter)
                                        self.y[numimage].append(ycenter)
                                        self.coordinatesToGroupNameAndBoxName[( xcenter, ycenter) ] = (group_folder, boxName)
                                        self.coordinatesToGroupNameAndImage[( xcenter, ycenter)] = group_folder + ", " + image


                    
                        self.kaverage[numimage].append(numboxes)
                        numimage += 1
    
    def calculateKMeans(self):
        for i in range(0, NUM_IMAGES_PER_GROUP):
            self.knum.append(round(statistics.mean(self.kaverage[i])))

            Data = {
                'x': self.x[i],
                'y': self.y[i]
            }
            df = DataFrame(Data,columns=['x','y'])
            self.kmeans = KMeans(n_clusters=self.knum[i])
            self.kmeans.fit(df)
            self.centroids.append(self.kmeans.cluster_centers_)
            self.labels.append(self.kmeans.labels_)

            self.plotKMeans(i, self.knum[i])


    def plotKMeans(self, i, knum):
        plt.plot(self.x[i], self.y[i], 'ko')

        for j in range(0, knum):
            plt.plot(self.centroids[i][j][0], self.centroids[i][j][1], 'ro')

        plt.title("cars00{}.xml".format(i + 1))
        plt.gca().invert_yaxis()
        plt.show()

    def calculateGroupScoresAndCollectBoxNamesForEachCluster(self):
        for i in range(0, NUM_IMAGES_PER_GROUP):
            GroupNamesToDistanceForImage = {}
            centroidToListOfBoxNames = {}

            # loop through every box from every group for a single image
            #print(self.labels[i])
            for j in range(0, len(self.x[i])):
                centroid = self.centroids[i][self.labels[i][j]]
                cx = centroid[0]
                cy = centroid[1]
                px = self.x[i][j]
                py = self.y[i][j]

                boxLabel = self.coordinatesToGroupNameAndBoxName[( px, py) ][1]

                if self.labels[i][j] in centroidToListOfBoxNames.keys():
                    centroidToListOfBoxNames[self.labels[i][j]].append(boxLabel)
                else:
                    centroidToListOfBoxNames[self.labels[i][j]] = [boxLabel]

                distance = math.sqrt( ((cx - px) ** 2) + ((cy - py) ** 2) )
                #print("{}, {} \t pt: ({},{}) \t distance:{} \t centroid: ({},{})".format(self.coordinatesToGroupNameAndBoxName[(px, py)][0], self.coordinatesToGroupNameAndBoxName[(px, py)][1], px, py, distance, cx, cy))
                
                groupName = self.coordinatesToGroupNameAndBoxName[(px, py)][0]
                if groupName in GroupNamesToDistanceForImage.keys():
                    GroupNamesToDistanceForImage[groupName] = GroupNamesToDistanceForImage[groupName] + distance
                else:
                    GroupNamesToDistanceForImage[groupName] = distance

            
            self.GroupNamesToDistanceList.append(GroupNamesToDistanceForImage)
            self.listOfDictionariesForCentroidsToBoxLabels.append(centroidToListOfBoxNames)

    def processBoxLabels(self):
        #loops through all images
        for i in range(0, NUM_IMAGES_PER_GROUP):
            centroidToListOfBoxNamesPopular = {}

            #loops through all clusters of an image
            print(i)
            for j in range(0, self.knum[i]):
                listOfBoxLabels = self.listOfDictionariesForCentroidsToBoxLabels[i][j]
                mostPopularLabel = self.most_frequent(listOfBoxLabels)

                centroidToListOfBoxNamesPopular[j] = mostPopularLabel

            self.listOfDictionariesForCentroidsToBoxLabelsPopular.append(centroidToListOfBoxNamesPopular)

    def printGroupScores(self):
        for i in range(0, NUM_IMAGES_PER_GROUP):
            print(i)
            print(self.listOfDictionariesForCentroidsToBoxLabels[i])
            print(self.GroupNamesToDistanceList[i])

    def calculateGroupScoresForBoxNames(self):
        for i in range(0, NUM_IMAGES_PER_GROUP):

            # loop through every box from every group for a single image
            print(self.x[i])
            #for j in range(0, len(self.x[i])):
                
                #centroid = self.centroids[i]
                #print(centroid)
  
                #self.listOfDictionariesForCentroidsToBoxLabelsPopular[i][()]
                #boxLabel = self.coordinatesToGroupNameAndBoxName[( px, py) ][1]
    

    def most_frequent(self, List): 
        return max(set(List), key = List.count) 


if __name__ == "__main__":
    dv = DataValidator()
    dv.initializeArrays()
    dv.readFiles()
    dv.calculateKMeans()
    dv.calculateGroupScoresAndCollectBoxNamesForEachCluster()
    dv.processBoxLabels()
    dv.calculateGroupScoresForBoxNames()

    #dv.printGroupScores()
    
