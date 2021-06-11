from PIL import Image, ImageOps
import numpy
'''
img_width=0
img_height = 0
px=0
'''
def startCmeans(c): #k is the number of clusters
	centroids_arr = []
	old_centroids = []
	i = 1
	while(1):
		counter = 0
		cond_break = True
		for increment in range(0, c):  #Initializes C centroids
			cent = px[numpy.random.randint(0, img_width), numpy.random.randint(0, img_height)]
			centroids_arr.append(cent)

		if all(x == (255,255,255) for x in centroids_arr):
			centroids_arr = [ ]
			cond_break = False
			pass

		else:
			for j in range(len(centroids_arr)):
				if centroids_arr[j] == (255,255,255):
					counter += 1

			if counter == 2:
				centroids_arr = [ ]
				cond_break = False
				pass
			
			else:
				cond_break = True

		if cond_break:
			break


	
	print("Centroids Initialized. Starting Assignments")
	print("===========================================")

	while not converged(centroids_arr, old_centroids) and i <= 20: # loop till we have smallest error, or 20 iterations
		print("Iteration #" + str(i))
		i += 1
		old_centroids = centroids_arr 								#Make the current centroids into the old centroids
		clusters = findCluster(centroids_arr) 						#Assign each pixel in the image to their respective centroids
		centroids_arr = updateCentroids(clusters) 	#Adjust the centroids to the center of their assigned pixels

	print("===========================================")
	print("Convergence Reached!")
	print(centroids_arr)
	return centroids_arr

def findCluster(centroids_arr):
	clusters = {}
	for x in range(0, img_width):
		for y in range(0, img_height):
			point = px[x, y]
			minIndex = getMinDist(px[x, y], centroids_arr)
			try: 
				clusters[minIndex].append(point)
			except KeyError:
				clusters[minIndex] = [point]

	return clusters

def updateCentroids(clusters):
	new_centroids = []
	keys = sorted(clusters.keys())

	for k in keys:
		n = numpy.mean(clusters[k], axis=0)
		new = (int(n[0]), int(n[1]), int(n[2]))
		print(str(k) + ": " + str(new))
		new_centroids.append(new)

	return new_centroids

def getMinDist(pixel, centroids):
	minDist = 9999
	minIndex = 0
	for i in range(0, len(centroids)):
		d = numpy.sqrt(int((centroids[i][0] - pixel[0]))**2 + int((centroids[i][1] - pixel[1]))**2 + int((centroids[i][2] - pixel[2]))**2)
		if d < minDist:
			minDist = d
			minIndex = i

	return minIndex

def converged(centroids_arr, old_centroids):
	if len(old_centroids) == 0: #
		return False

	#a= len(centroids_arr)/5 

	
	if len(centroids_arr) <= 5:
		a = 1
	elif len(centroids_arr) <= 10: 
		a = 2
	else:
		a = 4
	
       

	for i in range(0, len(centroids_arr)):
		cent = centroids_arr[i]
		old_cent = old_centroids[i]

		if not (((int(old_cent[0]) - a) <= cent[0] <= (int(old_cent[0]) + a)) and  #new centroid is within a certain range
            ((int(old_cent[1]) - a) <= cent[1] <= (int(old_cent[1]) + a)) and 
            ((int(old_cent[2]) - a) <= cent[2] <= (int(old_cent[2]) + a))):

			return False

	return True

def drawWindow(result_centroids):
	img = Image.new('RGB', (img_width, img_height))
	p = img.load() #RGB pixels of new image
	for x in range(img.size[0]):
		for y in range(img.size[1]):
			RGB_value = result_centroids[getMinDist(px[x, y], result_centroids)] #final assignment of pixel to RGB cluster
			p[x, y] = RGB_value
	#image=ImageOps.grayscale(img)
	#img.show()

def main(img,k_input):
	#im = Image.open(img)
	global img_width 
	global img_height
	img_width, img_height = img.size
	global px
	px = img.load()  #RGB data per pixel

	result_centroids = startCmeans(k_input)
	#drawWindow(result_centroids)
	return result_centroids
	drawWindow(result_centroids)

#im=Image.open('test11.bmp')
#main(im,3)

'''
im = Image.open(img)
img_width, img_height = im.size
px = im.load()  #RGB data per pixel

result_centroids = startKmeans(k_input)

drawWindow(result_centroids)
'''