import os, sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import json
# import subprocess
import pandas as pd

class WhiskerPadROI:
    def __init__(self, x, y, width, height, side):
        self.X = x
        self.Y = y
        self.Width = width
        self.Height = height
        self.ImageSide = side

class WhiskingFun:
    @staticmethod
    def find_whiskerpad(vid, splitUp=False):
        vidFrame = cv2.cvtColor(vid.read()[1], cv2.COLOR_BGR2GRAY)

        if splitUp is False:
            splitUp = "no"

        if splitUp.lower() == "no":
            # Get whisker pad coordinates
            whiskingParams = WhiskingFun.get_whisking_params(vidFrame)

        elif splitUp.lower() == "yes":
            nose_tip, face_axis, face_orientation = WhiskingFun.get_nose_tip_coordinates(vid.Path, vid.Name)
            midWidth = round(vidFrame.shape[1] / 2)
            if midWidth - vidFrame.shape[1] / 8 < nose_tip[0] < midWidth + vidFrame.shape[1] / 8:
                # If nosetip x value within +/- 1/8 of frame width, use that value
                midWidth = nose_tip[0]
            # Get whisking parameters for left side
            leftImage = vidFrame[:, :midWidth]
            whiskingParams = WhiskingFun.get_whisking_params(leftImage, midWidth - round(vidFrame.shape[1] / 2), face_axis, face_orientation)
            # Get whisking parameters for right side
            rightImage = vidFrame[:, midWidth:]
            whiskingParams[1] = WhiskingFun.get_whisking_params(rightImage, round(vidFrame.shape[1] / 2) - midWidth, face_axis, face_orientation)
            whiskingParams[0].ImageSide = 'Left'
            whiskingParams[1].ImageSide = 'Right'

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
            whiskingParams = WhiskingFun.get_whisking_params(leftImage, midWidth - round(vidFrame.shape[1] / 2), interactive=True)
            # Get whisking parameters for right side
            rightImage = vidFrame[:, midWidth:]
            whiskingParams[1] = WhiskingFun.get_whisking_params(rightImage, round(vidFrame.shape[1] / 2) - midWidth, interactive=True)
            whiskingParams[0].ImageSide = 'Left'
            whiskingParams[1].ImageSide = 'Right'

        return whiskingParams, splitUp

    @staticmethod
    def get_whisking_params(topviewImage, midlineOffset, nose_tip=None, face_axis=None, face_orientation=None, interactive=False):
        if interactive:
            wpCoordinates, wpLocation, wpRelativeLocation, sideBrightness = WhiskingFun.get_whiskerpad_coord_interactive(topviewImage)
        else:
            wpCoordinates, wpLocation, wpRelativeLocation, sideBrightness = WhiskingFun.get_whiskerpad_coord(topviewImage, nose_tip, face_axis, face_orientation)

        faceSideInImage, protractionDirection, linkingDirection = WhiskingFun.get_whiskerpad_params(wpCoordinates, wpRelativeLocation, sideBrightness)
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
    def get_whiskerpad_coord(topviewImage, nose_tip, face_axis, face_orientation):

        # TODO: debug this function
        
        # Get whisker pad coordinates, according to nose tip, face_axis and face orientation
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

        wpLocation = np.round([wpPosition[0] + wpPosition[2] / 2, wpPosition[1] + wpPosition[3] / 2])
        wpRelativeLocation = [wpLocation[0] / topviewImage.shape[1], wpLocation[1] / topviewImage.shape[0]]
        # Get vertices of the whisker pad ROI
        wpCoordinates = wpPosition + [wpPosition[0] + wpPosition[2], wpPosition[1] + wpPosition[3]]
        # wpCoordinates = np.round(wpAttributes.get_path().vertices)

        # Rotate image and ROI to get whisker-pad-centered image
        topviewImage_r = np.rot90(topviewImage, k=2)
        # np.rot90(topviewImage, k=int(wpAttributes.angle / 90))
        imageCenter = np.floor(wpLocation)
        # np.floor(wpAttributes.axes.camera_position[:2])
        center = np.tile(imageCenter, (wpCoordinates.shape[0], 1))
        theta = np.deg2rad(180)
        # np.deg2rad(wpAttributes.angle)
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
    def morph_open(image):
        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Crop image center to 400x400 pixels
        # find center of the image
        center_x, center_y = gray.shape[1]//2, gray.shape[0]//2
        # crop image
        crop_img = gray[center_y-200:center_y+200, center_x-200:center_x+200]
        
        # Apply an inverse binary threshold to the cropped image, with a threshold value of 9
        _, binary = cv2.threshold(crop_img, 9, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological opening to the thresholded image with anchor 5,5  1 iteration, shape rectangle 10,10
        kernel = np.ones((10,10),np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        return opening

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

            # Save the first frame as TIFF
            # cv2.imwrite("first_frame.tiff", image)

            # Threshold the first frame and apply morphological opening to the binary image
            opening = WhiskingFun.morph_open(image)

            # Find contours in the opened morphology
            contours = WhiskingFun.find_contours(opening)

            # Filter contours based on minimum area threshold
            minArea = 6000
            filteredContours = [cnt for cnt in contours if cv2.contourArea(cnt) >= minArea]

            # Find the largest contour
            contour = max(filteredContours, key=cv2.contourArea)

        # Close the video file
        vidcap.release()

        # Find the extreme points of the largest contour
        extrema = contour[:, 0, :]
        bottom_point = extrema[extrema[:, 1].argmax()]
        top_point = extrema[extrema[:, 1].argmin()]
        left_point = extrema[extrema[:, 0].argmin()]
        right_point = extrema[extrema[:, 0].argmax()]

        # We keep the extrema along the longest axis. 
        # The head side is the base of triangle, while the nose is the tip of the triangle
        if np.linalg.norm(bottom_point - top_point) > np.linalg.norm(left_point - right_point):
            # Find the base of the triangle: if left and right extrema are on the top half of the image, the top extrema is the base
            face_axis='vertical'
            if left_point[1] < 200 and right_point[1] < 200:
                face_orientation = 'up'
                nose_tip = top_point
            else:
                face_orientation = 'down'
                nose_tip = bottom_point
        else:
            # Find the base of the triangle: if top and bottom extrema are on the left half of the image, the left extrema is the base
            face_axis='horizontal'
            if top_point[0] < 200 and bottom_point[0] < 200:
                face_orientation = 'left'
                nose_tip = left_point
            else:
                face_orientation = 'right'
                nose_tip = right_point

        # Finally, adjust the nose tip coordinates to the original image coordinates
        nose_tip = nose_tip + np.array([200, 200])

        return nose_tip, face_axis, face_orientation

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("videofile", help="Path to the video file")
    parser.add_argument("--splitUp", action="store_true", help="Flag to split the video")
    parser.add_argument("--interactive", action="store_true", help="Flag for interactive mode")
    args = parser.parse_args()

    # Get whisking parameters
    if args.interactive:
        whiskingParams, splitUp = WhiskingFun.draw_whiskerpad_roi(args.videofile, args.splitUp)
    else:
        whiskingParams, splitUp = WhiskingFun.find_whiskerpad(args.videofile, args.splitUp)

    print(whiskingParams)
