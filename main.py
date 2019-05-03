import os, statistics, math, glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans


NUM_IMAGES_PER_GROUP = 2

class DataValidator():
    groupNames = []

    # given image number, list of x coordinates for every box center from every group
    x = []

    # given image number, list of y coordinates for every box center from every group
    y = []

    # this collects the number of boxes for each image per group, builds knum
    kaverage = []

    # given an image, contains "correct" number of boxes
    knum = []

    # given image number, list of centroid coordinates that correspond to box center from any group
    # in other words, we know which centroid each box belongs to
    centroids = []

    labels = []

    # given image number, dictionary of group name to total labeling errors
    groupNameToBoxScoreList = []

    # given image number, dictionary for centroid's coordinate to box label popular
    centroidsCoordinateToBoxLabel = []

    # given image number, given a coordinate, => group name and box name
    coordinatesToGroupNameAndBoxLabelsList = []

    # given image number, given a coordinate, => group name and image name
    coordinatesToGroupNameAndImageList = []

    # given image number, dictionary for group name to distance
    groupNameToDistanceList = []

    # given image number, dictionary for group name to numboxes
    groupNameToNumBoxesList = []

    def initializeArrays(self):

        for i in range(0, NUM_IMAGES_PER_GROUP):
            self.x.append([])
            self.y.append([])
            self.kaverage.append([])
            self.coordinatesToGroupNameAndBoxLabelsList.append({})
            self.coordinatesToGroupNameAndImageList.append({})
            self.groupNameToNumBoxesList.append({})


    def readFiles(self):
        testing_directory_path = os.path.expanduser("~/Desktop/TestThis4")
    
        group_folders = os.listdir(testing_directory_path)

        for group_folder in group_folders:
            group_path = os.path.join(testing_directory_path, group_folder)

            if (os.path.isdir(group_path)):
                if group_folder not in self.groupNames:
                    self.groupNames.append(group_folder)
                images = os.listdir(group_path)

                numimage = 0
                
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
                                        self.coordinatesToGroupNameAndBoxLabelsList[numimage][( xcenter, ycenter) ] = (group_folder, boxName)
                                        self.coordinatesToGroupNameAndImageList[numimage][( xcenter, ycenter)] = group_folder + ", " + image

                        self.kaverage[numimage].append(numboxes)
                        self.groupNameToNumBoxesList[numimage][group_folder] = numboxes
                        numimage += 1
    
    def calculateKNum(self):
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
            #self.plotKMeans(i, self.knum[i])

    def plotKMeans(self, i, knum):
        plt.plot(self.x[i], self.y[i], 'ko')

        for j in range(0, knum):
            plt.plot(self.centroids[i][j][0], self.centroids[i][j][1], 'ro')

        plt.title("cars00{}.xml".format(i + 1))
        plt.gca().invert_yaxis()
        plt.show()

    def calculateGroupScoresAndBoxNamePerCluster(self):
        for i in range(0, NUM_IMAGES_PER_GROUP):
            groupNamesToDistance = {}
            centroidToListOfBoxNames = {}

            # loop through every box from every group for a single image
            for j in range(0, len(self.x[i])):
                centroid = self.centroids[i][self.labels[i][j]]
                cx = centroid[0]
                cy = centroid[1]
                px = self.x[i][j]
                py = self.y[i][j]

                boxLabel = self.coordinatesToGroupNameAndBoxLabelsList[i][( px, py) ][1]

                if (cx, cy) in centroidToListOfBoxNames.keys():
                    centroidToListOfBoxNames[(cx, cy)].append(boxLabel)
                else:
                    centroidToListOfBoxNames[(cx, cy)] = [boxLabel]

                distance = math.sqrt( ((cx - px) ** 2) + ((cy - py) ** 2) )
                
                groupName = self.coordinatesToGroupNameAndBoxLabelsList[i][(px, py)][0]
                if groupName in groupNamesToDistance.keys():
                    groupNamesToDistance[groupName] = groupNamesToDistance[groupName] + distance
                else:
                    groupNamesToDistance[groupName] = distance

            for key, value in centroidToListOfBoxNames.items():
                centroidToListOfBoxNames[key] = self.most_frequent(value)

            self.groupNameToDistanceList.append(groupNamesToDistance)
            self.centroidsCoordinateToBoxLabel.append(centroidToListOfBoxNames)

    def debug(self):
        for i in range(0, NUM_IMAGES_PER_GROUP):
            print("image: {}".format(i))
            print("Labeling Errors Score: {}".format(self.groupNameToBoxScoreList[i]))
            print("Box Placement Score: {}".format(self.groupNameToDistanceList[i]))
            print("Num Boxes: {} => {}".format(self.groupNameToNumBoxesList[i], self.knum[i]))
            print("")

    def graphLabelingErrorsForOneImage(self, i):
        self.groupNames.sort()
        score = []

        for groupName in self.groupNames:
            score.append(self.groupNameToBoxScoreList[i][groupName])

        index = np.arange(len(self.groupNames))
        plt.bar(index, score)
        plt.xlabel('Group Names', fontsize=10)
        plt.ylabel('No of Errors', fontsize=10)
        plt.xticks(index, self.groupNames, fontsize=10, rotation=30)
        plt.title("Labeling Errors for Image {}".format(i))
        plt.show()

    def graphDistanceScoreForOneImage(self, i):
        self.groupNames.sort()
        score = []

        for groupName in self.groupNames:
            score.append(self.groupNameToDistanceList[i][groupName])

        index = np.arange(len(self.groupNames))
        plt.bar(index, score)
        plt.xlabel('Group Names', fontsize=10)
        plt.ylabel('Distances', fontsize=10)
        plt.xticks(index, self.groupNames, fontsize=10, rotation=30)
        plt.title("Distances for Image {}".format(i))
        plt.show()

    def graphDistanceScoreTotal(self):
        self.groupNames.sort()
        score = []




        index = np.arange(len(self.groupNames))
        plt.bar(index, score)
        plt.xlabel('Group Names', fontsize=10)
        plt.ylabel('Distances', fontsize=10)
        plt.xticks(index, self.groupNames, fontsize=10, rotation=30)
        plt.title("Total Distances")
        plt.show()


    def calculateGroupScoresForBoxNames(self):
        for i in range(0, NUM_IMAGES_PER_GROUP):
            groupnameToBoxLabelErrorsScore = {}
            
            for j in range(0, len(self.x[i])):
                centroid = self.centroids[i][self.labels[i][j]]
                cx = centroid[0]
                cy = centroid[1]
                px = self.x[i][j]
                py = self.y[i][j]
                groupName = self.coordinatesToGroupNameAndBoxLabelsList[i][(px, py)][0]
                boxname = self.coordinatesToGroupNameAndBoxLabelsList[i][(px, py)][1]
                correctBoxName = self.centroidsCoordinateToBoxLabel[i][(cx, cy)]

                if groupName in groupnameToBoxLabelErrorsScore.keys():
                    if (boxname != correctBoxName):
                        groupnameToBoxLabelErrorsScore[groupName] = groupnameToBoxLabelErrorsScore[groupName] + 1
                else:
                    if (boxname != correctBoxName):
                        groupnameToBoxLabelErrorsScore[groupName] = 1
                    else:
                        groupnameToBoxLabelErrorsScore[groupName] = 0

            self.groupNameToBoxScoreList.append(groupnameToBoxLabelErrorsScore)
    

    def most_frequent(self, List): 
        return max(set(List), key = List.count) 


if __name__ == "__main__":
    dv = DataValidator()
    dv.initializeArrays()
    dv.readFiles()
    dv.calculateKNum()
    dv.calculateGroupScoresAndBoxNamePerCluster()
    dv.calculateGroupScoresForBoxNames()
    dv.debug()

    dv.graphLabelingErrorsForOneImage(0)
    dv.graphDistanceScoreForOneImage(0)
    #dv.graphDistanceScoreTotal()


