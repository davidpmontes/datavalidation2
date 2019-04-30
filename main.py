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
    x = []
    y = []
    kaverage = []
    centroids = []
    labels = []
    coordinatesToGroupName = {}
    coordinatesToGroupNameAndImage = {}

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
                                numboxes += 1
                                for childp in child:
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
                                        self.coordinatesToGroupName[( xcenter, ycenter) ] = group_folder
                                        self.coordinatesToGroupNameAndImage[( xcenter, ycenter)] = group_folder + ", " + image


                    
                        #print("filename: {}, numimage: {}, numboxes: {}".format(filename, numimage, numboxes))
                        self.kaverage[numimage].append(numboxes)
                        numimage += 1
        #print(self.kaverage)
        #print(self.x)
        #print(self.y)
    
    def calculateKMeans(self):
        for i in range(0, NUM_IMAGES_PER_GROUP):
            #print(self.kaverage[i])
            knum = round(statistics.mean(self.kaverage[i]))
            #print(knum)

            Data = {
                'x': self.x[i],
                'y': self.y[i]
            }
            df = DataFrame(Data,columns=['x','y'])
            self.kmeans = KMeans(n_clusters=knum)
            self.kmeans.fit(df)

            self.centroids.append(self.kmeans.cluster_centers_)
            print(self.kmeans.cluster_centers_)

            #print(self.x)
            #print(self.y)
            #print(self.centroids)
            self.labels.append(self.kmeans.labels_)
            print(self.kmeans.labels_)

            #print(self.dictionary)

            plt.plot(self.x[i], self.y[i], 'ko')

            for j in range(0, knum):
                plt.plot(self.centroids[i][j][0], self.centroids[i][j][1], 'ro')

            plt.title("cars00{}.xml".format(i + 1))
            plt.gca().invert_yaxis()
            plt.show()

    def giveScore(self):
        for i in range(0, NUM_IMAGES_PER_GROUP):
            for j in range(0, len(self.x[i])):
                centroid = self.centroids[i][self.labels[i][j]]
                cx = centroid[0]
                cy = centroid[1]
                px = self.x[i][j]
                py = self.y[i][j]

                # print(centroid[0])
                # print(centroid[1])
                # print(self.x[i][j])
                # print(self.y[i][j])
                distance = math.sqrt( ((cx - px) ** 2) + ((cy - py) ** 2) )
                print("{} \t pt: ({},{}) \t distance:{} \t centroid: ({},{})".format(self.coordinatesToGroupNameAndImage[(px, py)], px, py, distance, cx, cy))

if __name__ == "__main__":
    dv = DataValidator()
    dv.initializeArrays()
    dv.readFiles()
    dv.calculateKMeans()
    dv.giveScore()
    # print(dv.centroids)