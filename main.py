import os
from pandas import DataFrame
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
import statistics
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



NUM_IMAGES_PER_GROUP = 2

class DataValidator():
    x = []
    y = []
    kaverage = []

    def initializeArrays(self):

        for i in range(0, NUM_IMAGES_PER_GROUP):
            self.x.append([])
            self.y.append([])
            self.kaverage.append([])

    def readFiles(self):
        testing_directory_path = os.path.expanduser("~/Desktop/TestThis")
    
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
                                        self.x[numimage].append(xtotal / 4)
                                        self.y[numimage].append(ytotal / 4)

                    
                        #print("filename: {}, numimage: {}, numboxes: {}".format(filename, numimage, numboxes))
                        self.kaverage[numimage].append(numboxes)
                        numimage += 1
        #print(self.kaverage)
        #print(self.x)
        #print(self.y)
    
    def calculateKMeans(self):
        for i in range(0, NUM_IMAGES_PER_GROUP):
            knum = round(statistics.mean(self.kaverage[i]))

            Data = {
                'x': self.x[i],
                'y': self.y[i]
            }
            df = DataFrame(Data,columns=['x','y'])
            kmeans = KMeans(n_clusters=knum).fit(df)
            centroids = kmeans.cluster_centers_

            plt.plot(self.x[i], self.y[i], 'ko')

            for j in range(0, knum):
                plt.plot(centroids[j][0], centroids[j][1], 'ro')

            plt.title("cars00{}.xml".format(i + 1))
            plt.show()


if __name__ == "__main__":
    dv = DataValidator()
    dv.initializeArrays()
    dv.readFiles()
    dv.calculateKMeans()