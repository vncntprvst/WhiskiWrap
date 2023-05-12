import os, sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import json
# import subprocess
import pandas as pd

video_dir = ""

class WhiskerPadROI:
    def __init__(self, x, y, width, height, side):
        self.X = x
        self.Y = y
        self.Width = width
        self.Height = height
        self.ImageSide = side

class WhiskingFun:
    @staticmethod
    def find_whiskerpad(videofile, splitUp=False):
        cap = cv2.VideoCapture(videofile)
        vidFrame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
        cap.release()
        if not splitUp:
            # Get whisker pad coordinates
            whiskingParams = WhiskingFun.get_whisking_params(vidFrame)

        elif splitUp:
            nose_tip, face_axis, face_orientation = WhiskingFun.get_nose_tip_coordinates(videofile)
            if face_axis == 'vertical':
                midWidth = round(vidFrame.shape[1] / 2)
                if midWidth - vidFrame.shape[1] / 8 < nose_tip[0] < midWidth + vidFrame.shape[1] / 8:
                    # If nosetip x value within +/- 1/8 of frame width, use that value
                    midWidth = nose_tip[0]
                # Get whisking parameters for left side
                leftImage = vidFrame[:, :midWidth]
                whiskingParams = WhiskingFun.get_whisking_params(leftImage, midWidth - round(vidFrame.shape[1] / 2), nose_tip, face_axis, face_orientation, image_side='Left')
                # Get whisking parameters for right side
                rightImage = vidFrame[:, midWidth:]
                whiskingParams[1] = WhiskingFun.get_whisking_params(rightImage, round(vidFrame.shape[1] / 2) - midWidth, nose_tip, face_axis, face_orientation, image_side='Right')
                whiskingParams[0].ImageSide = 'Left'
                whiskingParams[1].ImageSide = 'Right'

            elif face_axis == 'horizontal':
                midWidth = round(vidFrame.shape[0] / 2)
                if midWidth - vidFrame.shape[0] / 8 < nose_tip[1] < midWidth + vidFrame.shape[0] / 8:
                    # If nosetip y value within +/- 1/8 of frame height, use that value
                    midWidth = nose_tip[1]
                # Get whisking parameters for top side
                topImage = vidFrame[:midWidth, :]
                whiskingParams = WhiskingFun.get_whisking_params(topImage, midWidth - round(vidFrame.shape[0] / 2), nose_tip, face_axis, face_orientation, image_side='Top')
                # Get whisking parameters for bottom side
                bottomImage = vidFrame[midWidth:, :]
                whiskingParams[1] = WhiskingFun.get_whisking_params(bottomImage, round(vidFrame.shape[0] / 2) - midWidth, nose_tip, face_axis, face_orientation, image_side='Bottom')
                whiskingParams[0].ImageSide = 'Top'
                whiskingParams[1].ImageSide = 'Bottom'

        return whiskingParams, splitUp

    @staticmethod
    def draw_whiskerpad_roi(vid, splitUp=None):
        vidFrame = cv2.cvtColor(vid.read()[1], cv2.COLOR_BGR2GRAY)
        cv2.imshow("Video Frame", vidFrame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if splitUp is None:
            splitUp = input("Do you need to split the video? (Yes/No): ")

        if splitUp.lower() == "no":
            # Get whisker pad coordinates
            # Get whisking parameters
            whiskingParams = WhiskingFun.get_whisking_params(vidFrame,interactive=True)
            # Clear variables firstVideo
        elif splitUp.lower() == "yes":
            nose_tip = WhiskingFun.get_nose_tip_coordinates(vid.Path, vid.Name)
            midWidth = round(vidFrame.shape[1] / 2)
            if midWidth - vidFrame.shape[1] / 8 < nose_tip[0] < midWidth + vidFrame.shape[1] / 8:
                # If nosetip x value within +/- 1/8 of frame width, use that value
                midWidth = nose_tip[0]
            # Get whisking parameters for left side
            leftImage = vidFrame[:, :midWidth]
            whiskingParams = WhiskingFun.get_whisking_params(leftImage, midWidth - round(vidFrame.shape[1] / 2), image_side='Left', interactive=True)
            # Get whisking parameters for right side
            rightImage = vidFrame[:, midWidth:]
            whiskingParams[1] = WhiskingFun.get_whisking_params(rightImage, round(vidFrame.shape[1] / 2) - midWidth, image_side='Right', interactive=True)
            whiskingParams[0].ImageSide = 'Left'
            whiskingParams[1].ImageSide = 'Right'

        return whiskingParams, splitUp

    @staticmethod
    def get_whisking_params(topviewImage, midlineOffset, nose_tip=None, face_axis=None, face_orientation=None, image_side=None, interactive=False):
        if interactive:
            wpCoordinates, wpLocation, wpRelativeLocation = WhiskingFun.get_whiskerpad_coord_interactive(topviewImage)
        else:
            wpCoordinates, wpLocation, wpRelativeLocation = WhiskingFun.get_whiskerpad_coord(topviewImage, nose_tip, face_axis, face_orientation, image_side)

        faceSideInImage, protractionDirection, linkingDirection = WhiskingFun.get_whiskerpad_params(wpCoordinates, wpRelativeLocation)
        whiskingParams = {
            'Coordinates': np.round(wpCoordinates, 2),
            'Location': wpLocation,
            'RelativeLocation': np.round(wpRelativeLocation, 2),
            'FaceSideInImage': faceSideInImage,
            'ProtractionDirection': protractionDirection,
            'LinkingDirection': linkingDirection,
            'MidlineOffset': midlineOffset,
            'ImageDimensions': topviewImage.shape
        }
        return whiskingParams

    @staticmethod
    def get_whiskerpad_coord(topviewImage, nose_tip, face_axis, face_orientation, image_side):

        # TODO: debug this function
        
        contour=None
        while contour is None or len(contour) == 0:
            # Threshold the first frame and apply morphological opening to the binary image
            gray, opening = WhiskingFun.morph_open(topviewImage)

            # Find contours in the opened morphology
            contours = WhiskingFun.find_contours(opening)

            # Filter contours based on minimum area threshold
            minArea = 3000
            filteredContours = [cnt for cnt in contours if cv2.contourArea(cnt) >= minArea]

            # Find the largest contour
            contour = max(filteredContours, key=cv2.contourArea)

        # plot image, and overlay the contour on top
        # fig, ax = plt.subplots()
        # ax.imshow(topviewImage)
        # plt.title('Face contour')
        # ax.plot(contour[:, 0, 0], contour[:, 0, 1], linewidth=2, color='r')
        # plt.show()

        # At this point, the contour is roughly a right triangle: 
        # two straight sides are the image border, and the face contour is the "hypothenuse".
        # Extract the face outline from the contour.

        if face_axis == 'vertical':
            if face_orientation == 'down':
                # starting point is the point with the lowest x value and lowest y value
                starting_point = contour[contour[:, 0, 1].argmin(), 0, :]
                # Find the index of the starting point in the contour
                starting_point_index = np.where((contour == starting_point).all(axis=2))[0][0]
                # ending point is the point with the highest x value and highest y value
                ending_point = contour[contour[:, 0, 1].argmax(), 0, :]
                # Find the index of the ending point in the contour
                ending_point_index = np.where((contour == ending_point).all(axis=2))[0][0]

                # face contour is the part of the contour bounded by those indices
                face_contour = contour[starting_point_index:ending_point_index+1, 0, :]

            elif face_orientation == 'up':
                # starting point is the point with the lowest x value and highest y value
                starting_point = contour[contour[:, 0, 1].argmax(), 0, :]
                # Find the index of the starting point in the contour
                starting_point_index = np.where((contour == starting_point).all(axis=2))[0][0]
                # ending point is the point with the highest x value and lowest y value
                ending_point = contour[contour[:, 0, 1].argmin(), 0, :]
                # Find the index of the ending point in the contour
                ending_point_index = np.where((contour == ending_point).all(axis=2))[0][0]

                # face contour is the part of the contour bounded by those indices
                face_contour = contour[starting_point_index:ending_point_index+1, 0, :]
        elif face_axis == 'horizontal':
            if face_orientation == 'left':
                # starting point is the point with the lowest x value and lowest y value
                starting_point = contour[contour[:, 0, 0].argmin(), 0, :]
                # Find the index of the starting point in the contour
                starting_point_index = np.where((contour == starting_point).all(axis=2))[0][0]
                # ending point is the point with the highest x value and highest y value
                ending_point = contour[contour[:, 0, 0].argmax(), 0, :]
                # Find the index of the ending point in the contour
                ending_point_index = np.where((contour == ending_point).all(axis=2))[0][0]

                # face contour is the part of the contour bounded by those indices
                face_contour = contour[starting_point_index:ending_point_index+1, 0, :]

            elif face_orientation == 'right':
                # starting point is the point with the highest x value and lowest y value
                starting_point = contour[contour[:, 0, 0].argmax(), 0, :]
                # Find the index of the starting point in the contour
                starting_point_index = np.where((contour == starting_point).all(axis=2))[0][0]
                # ending point is the point with the lowest x value and highest y value
                ending_point = contour[contour[:, 0, 0].argmin(), 0, :]
                # Find the index of the ending point in the contour
                ending_point_index = np.where((contour == ending_point).all(axis=2))[0][0]

                # face contour is the part of the contour bounded by those indices
                face_contour = contour[starting_point_index:ending_point_index+1, 0, :]

        # now if a straight line between the starting point and the ending point
        # and rotate it to plot it as the x-axis, let's find the highest y value 
        # on the rotated contour, and use that as the whisker pad location
        # first, find the angle of the straight line
        theta = np.arctan((ending_point[1] - starting_point[1]) / (ending_point[0] - starting_point[0]))
        # then, rotate the contour by that angle
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        face_contour_r = np.round(np.dot(face_contour - starting_point, rot) + starting_point)

        # # PLot the face contour on top of the image
        # fig, ax = plt.subplots()
        # ax.imshow(topviewImage)
        # plt.title('Face contour')
        # ax.plot(face_contour[:, 0], face_contour[:, 1], linewidth=2, color='r')
        # plt.show()

        # Mext, find the index of the highest y value on the rotated contour
        wpLocationIndex = face_contour_r[:, 1].argmax()
        # the whisker pad location in the original image is the wpLocationIndex of the face contour 
        wpLocation = face_contour[wpLocationIndex, :]

        # # Plot image and wpLocation on top
        # fig, ax = plt.subplots()
        # ax.imshow(topviewImage)
        # plt.title('Face contour')
        # ax.plot(face_contour[:, 0], face_contour[:, 1], linewidth=2, color='r')
        # ax.plot(wpLocation[0], wpLocation[1], 'o', color='y')
        # plt.show()

        # if is within 200 pixels of the nose tip, keep it
        if np.linalg.norm(wpLocation - nose_tip) < 200:

            # Save the image with the contour overlayed and the whisker pad location labelled on it
            image_with_contour = topviewImage.copy()
            cv2.drawContours(image_with_contour, [face_contour], -1, (0, 255, 0), 3)
            # define file name based on face orientation 
            output_path = os.path.join(video_dir, 'whiskerpad_' + image_side.lower() + '.jpg')
            cv2.imwrite(output_path, cv2.circle(image_with_contour, tuple(wpLocation), 10, (255, 0, 0), -1))

            # Define whisker pad position (wpPosition) as the rectangle around the whisker pad location
            wpPosition = [wpLocation[0] - 10, wpLocation[1] - 10, 20, 20]

        else:
            # Get ballpark whisker pad coordinates, according to nose tip, face_axis and face orientation
            # Define whiskerpad position (wpPosition) as the rectangle around the nose tip, offset by 10 pixels towards the face
            if face_axis == 'vertical':
                if face_orientation == 'up':
                    wpPosition = [nose_tip[0] - 10, nose_tip[1] + 10, 20, 20]
                elif face_orientation == 'down':
                    wpPosition = [nose_tip[0] - 10, nose_tip[1] - 10, 20, 20]
            elif face_axis == 'horizontal':
                if face_orientation == 'left':
                    wpPosition = [nose_tip[0] + 10, nose_tip[1] - 10, 20, 20]
                elif face_orientation == 'right':
                    wpPosition = [nose_tip[0] - 10, nose_tip[1] - 10, 20, 20]

            # Define whisker pad location (wpLocation) as the center of the whisker pad position
            wpLocation = np.round([wpPosition[0] + wpPosition[2] / 2, wpPosition[1] + wpPosition[3] / 2])

        # Define whisker pad relative location (wpRelativeLocation) as the whisker pad location divided by the image dimensions
        wpRelativeLocation = [wpLocation[0] / topviewImage.shape[1], wpLocation[1] / topviewImage.shape[0]]

        # plot image, and overlay whisker pad position and location on top
        # fig, ax = plt.subplots()
        # ax.imshow(topviewImage)
        # plt.title('Draw rectangle around whisker pad')
        # wpAttributes = plt.Rectangle((wpPosition[0], wpPosition[1]), wpPosition[2], wpPosition[3], label='align', visible=False, fill=False)
        # ax.add_patch(wpAttributes)
        # plt.show()

        return wpPosition, wpLocation, wpRelativeLocation

    @staticmethod
    def get_side_brightness(wpImage):
        # Find brightness for each side
        # top_bottom_ratio = np.sum(wpImage[0, :]) / np.sum(wpImage[-1, :])
        # left_right_ratio = np.sum(wpImage[:, 0]) / np.sum(wpImage[:, -1])
        top_brightness=np.sum(wpImage[0, :])
        bottom_brightness=np.sum(wpImage[-1, :])
        left_brightness=np.sum(wpImage[:, 0])
        right_brightness=np.sum(wpImage[:, -1])

        sideBrightness = {
            'top': top_brightness,
            'bottom': bottom_brightness,
            'left': left_brightness,
            'right': right_brightness
        }

        # assign label to the side with the highest brightness
        sideBrightness['maxSide']=max(sideBrightness, key=sideBrightness.get) 
    
        return sideBrightness

    @staticmethod
    def get_whiskerpad_coord_interactive(topviewImage):
        fig, ax = plt.subplots()
        ax.imshow(topviewImage)
        plt.title('Draw rectangle around whisker pad')
        wpAttributes = plt.Rectangle((0, 0), 1, 1, label='align', visible=False, fill=False)
        ax.add_patch(wpAttributes)
        plt.show()

        wpPosition = wpAttributes.get_bbox().bounds
        wpLocation = np.round([wpPosition[0] + wpPosition[2] / 2, wpPosition[1] + wpPosition[3] / 2])
        wpRelativeLocation = [wpLocation[0] / topviewImage.shape[1], wpLocation[1] / topviewImage.shape[0]]
        wpCoordinates = np.round(wpAttributes.get_path().vertices)

        # Rotate image and ROI to get whisker-pad-centered image
        topviewImage_r = np.rot90(topviewImage, k=int(wpAttributes.angle / 90))
        imageCenter = np.floor(wpAttributes.axes.camera_position[:2])
        center = np.tile(imageCenter, (wpCoordinates.shape[0], 1))
        theta = np.deg2rad(wpAttributes.angle)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        wpCoordinates_r = np.round(np.dot((wpCoordinates - center), rot) + center)
        wpCoordinates_r[wpCoordinates_r <= 0] = 1

        wpImage = topviewImage_r[
            wpCoordinates_r[0, 1]:wpCoordinates_r[1, 1], wpCoordinates_r[1, 0]:wpCoordinates_r[2, 0], 0]

        # Find brightness ratio for each dimension
        top_bottom_ratio = np.sum(wpImage[0, :]) / np.sum(wpImage[-1, :])
        left_right_ratio = np.sum(wpImage[:, 0]) / np.sum(wpImage[:, -1])

        sideBrightness = {
            'top_bottom_ratio': top_bottom_ratio,
            'left_right_ratio': left_right_ratio
        }

        return wpCoordinates, wpLocation, wpRelativeLocation, sideBrightness

    @staticmethod
    def get_whiskerpad_params(wpCoordinates, wpRelativeLocation, sideBrightness):
        is_horizontal = np.linalg.norm(wpCoordinates[0] - wpCoordinates[1]) / np.linalg.norm(wpCoordinates[1] - wpCoordinates[2]) < 1

        if is_horizontal:
            faceSideInImage = 'bottom' if sideBrightness['top_bottom_ratio'] > 1 else 'top'
            protractionDirection = 'leftward' if sideBrightness['left_right_ratio'] > 1 else 'rightward'
        else:
            faceSideInImage = 'right' if sideBrightness['left_right_ratio'] > 1 else 'left'
            protractionDirection = 'upward' if sideBrightness['top_bottom_ratio'] > 1 else 'downward'

        linkingDirection = 'rostral'

        return faceSideInImage, protractionDirection, linkingDirection

    @staticmethod
    def save_whiskerpad_params(whiskingParams, trackingDir):
        with open(os.path.join(trackingDir, 'whiskerpad.json'), 'w') as file:
            json.dump(whiskingParams, file, indent='\t')

    @staticmethod
    def RestrictToWhiskerPad(wData, whiskerpadCoords, ImageDim):
        if len(whiskerpadCoords) == 4:  # Simple rectangular ROI x, y, width, height
            blacklist = (
                (wData['follicle_x'] > whiskerpadCoords[0] + whiskerpadCoords[2]) |
                (wData['follicle_x'] < whiskerpadCoords[0]) |
                (wData['follicle_y'] > whiskerpadCoords[1] + whiskerpadCoords[3]) |
                (wData['follicle_y'] < whiskerpadCoords[1])
            )
        elif len(whiskerpadCoords) >= 8:  # ROI Vertices (x, y)n
            wpPath = Path(whiskerpadCoords)
            follPoints = np.column_stack((wData['follicle_x'], wData['follicle_y']))
            blacklist = ~wpPath.contains_points(follPoints)
        else:
            return wData, []

        blacklist = blacklist.ravel()
        wData = wData[~blacklist]
        return wData, blacklist.tolist()

    @staticmethod
    def morph_open(image, crop=False):
        # convert to grayscale
        if image.ndim == 3 and image.shape[2] == 3:  # Check if image has 3 dimensions and 3 channels (BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # No need for conversion, already grayscale

        if crop:
            # Crop image center to 400x400 pixels
            # find center of the image
            center_x, center_y = gray.shape[1]//2, gray.shape[0]//2
            # crop image
            gray = gray[center_y-200:center_y+200, center_x-200:center_x+200]
            
        # Apply an inverse binary threshold to the cropped image, with a threshold value of 9
        _, binary = cv2.threshold(gray, 9, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological opening to the thresholded image with anchor 5,5  1 iteration, shape rectangle 10,10
        kernel = np.ones((10,10),np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        return gray, opening

    @staticmethod
    def find_contours(opening):
        # Find contours in the opened image with method CHAIN_APPROX_NONE, mode external, offset 0,0
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return contours

    @staticmethod
    def get_nose_tip_coordinates(videoFileName):
        ## Using OpenCV

        # Open video file defined by videoDirectory, videoFileName 
        # Open the video file
        vidcap = cv2.VideoCapture(videoFileName)
        
        contour=None
        while contour is None or len(contour) == 0:
            # open the next frame
            _, image = vidcap.read()

            # Threshold the first frame and apply morphological opening to the binary image
            gray, opening = WhiskingFun.morph_open(image, crop=True)

            # Find contours in the opened morphology
            contours = WhiskingFun.find_contours(opening)

            # Filter contours based on minimum area threshold
            minArea = 6000
            filteredContours = [cnt for cnt in contours if cv2.contourArea(cnt) >= minArea]

            # Find the largest contour
            contour = max(filteredContours, key=cv2.contourArea)

        # Close the video file
        vidcap.release()

        sideBrightness=WhiskingFun.get_side_brightness(opening)

        # Find the extreme points of the largest contour
        extrema = contour[:, 0, :]
        bottom_point = extrema[extrema[:, 1].argmax()]
        top_point = extrema[extrema[:, 1].argmin()]
        left_point = extrema[extrema[:, 0].argmin()]
        right_point = extrema[extrema[:, 0].argmax()]

        # We keep the extrema along the longest axis. 
        # The head side is the base of triangle, while the nose is the tip of the triangle
        if sideBrightness['maxSide'] == 'top' or sideBrightness['maxSide'] == 'bottom':
            # Find the base of the triangle: if left and right extrema are on the top half of the image, the top extrema is the base
            face_axis='vertical'
            if sideBrightness['maxSide'] == 'bottom':
                face_orientation = 'up'
                nose_tip = top_point
            else:
                face_orientation = 'down'
                nose_tip = bottom_point
        else:
            # Find the base of the triangle: if top and bottom extrema are on the left half of the image, the left extrema is the base
            face_axis='horizontal'
            if sideBrightness['maxSide'] == 'right':
                face_orientation = 'left'
                nose_tip = left_point
            else:
                face_orientation = 'right'
                nose_tip = right_point

        # Finally, adjust the nose tip coordinates to the original image coordinates
        nose_tip = nose_tip + np.array([gray.shape[1]//2-200, gray.shape[0]//2-200])

        # Save the frame with the nose tip labelled on it
        cv2.imwrite(os.path.join(video_dir, 'nose_tip.jpg'), cv2.circle(image, tuple(nose_tip), 10, (255, 0, 0), -1))

        return nose_tip, face_axis, face_orientation

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("videofile", help="Path to the video file")
    parser.add_argument("--splitUp", action="store_true", help="Flag to split the video")
    parser.add_argument("--interactive", action="store_true", help="Flag for interactive mode")
    args = parser.parse_args()

    # assign value to global variable video_dir
    video_dir = os.path.dirname(args.videofile)

    # Get whisking parameters
    if args.interactive:
        whiskingParams, splitUp = WhiskingFun.draw_whiskerpad_roi(args.videofile, args.splitUp)
    else:
        whiskingParams, splitUp = WhiskingFun.find_whiskerpad(args.videofile, args.splitUp)

    print(whiskingParams)
