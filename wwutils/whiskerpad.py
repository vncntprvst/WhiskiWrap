"""whiskerpad.py
Methods for extracting whisker pad coordinates from video files

Part of the Whisker Tracking pipeline for analyzing rodent whisker data.
Will likely be removed from this repository in the future.

Author: Vincent Prevosto <prevosto@mit.edu>  
Date  : 05/19/2023

Use is subject to the MIT License (MIT)
"""

import os, sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import json
# import subprocess
import pandas as pd
import copy

video_dir = ""

class WhiskingParams:
    def __init__(self, wpArea, wpLocation, wpRelativeLocation, fp=None):
        if fp is not None:
            self.FaceAxis = fp.FaceAxis
            self.FaceOrientation = fp.FaceOrientation
            self.FaceSide = fp.FaceSide
            self.NoseTip = fp.NoseTip
            self.MidlineOffset = fp.MidlineOffset
        self.AreaCoordinates = wpArea
        self.Location = wpLocation
        self.RelativeLocation = wpRelativeLocation
        self.ImageSide = None
        self.ImageBorderAxis = None
        self.ProtractionDirection = None
        self.LinkingDirection = None
        self.ImageDimensions = []

class WhiskingParamsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, WhiskingParams):
            # Convert WhiskingParams object to a dictionary
            params_dict = obj.__dict__.copy()
            # Convert ndarray objects to lists
            params_dict['Location'] = params_dict['Location'].tolist()
            params_dict['NoseTip'] = params_dict['NoseTip'].tolist()
            # Convert numpy.int64 to int
            params_dict['MidlineOffset'] = int(params_dict['MidlineOffset'])
            # Return the updated dictionary
            return params_dict
        if isinstance(obj, np.int64):
            # Convert numpy.int64 to int
            return int(obj)
        return super().default(obj)

class FaceParams:
    def __init__(self, face_axis, face_orientation, face_side=None, nose_tip=None, midline_offset=0):
        self.FaceAxis = face_axis
        self.FaceOrientation = face_orientation
        self.FaceSide = face_side
        self.NoseTip = nose_tip
        self.MidlineOffset = midline_offset
    
    def copy(self):
        return copy.deepcopy(self)

class WhiskingFun:
    @staticmethod
    def get_whiskerpad_params(videofile, splitUp=False):
        fp, initialFrame = WhiskingFun.get_nose_tip_coordinates(videofile)
        vidcap = cv2.VideoCapture(videofile)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, initialFrame)
        vidFrame = cv2.cvtColor(vidcap.read()[1], cv2.COLOR_BGR2GRAY)
        vidcap.release()

        if splitUp:
            if fp.FaceAxis == 'vertical':
                midWidth = round(vidFrame.shape[1] / 2)
                if midWidth - vidFrame.shape[1] / 8 < fp.NoseTip[0] < midWidth + vidFrame.shape[1] / 8:
                    # If nosetip x value within +/- 1/8 of frame width, use that value
                    midWidth = fp.NoseTip[0]
                
                # Broadcast image two halves into two arrays
                if fp.FaceOrientation == 'up':
                    image_halves = [vidFrame[:, :midWidth], vidFrame[:, midWidth:]]
                    image_side = ['Left', 'Right']
                    
                elif fp.FaceOrientation == 'down':
                    image_halves = [vidFrame[:, midWidth:], vidFrame[:, :midWidth]]
                    image_side = ['Right', 'Left']

            elif fp.FaceAxis == 'horizontal':
                midWidth = round(vidFrame.shape[0] / 2)
                if midWidth - vidFrame.shape[0] / 8 < fp.NoseTip[1] < midWidth + vidFrame.shape[0] / 8:
                    # If nosetip y value within +/- 1/8 of frame height, use that value
                    midWidth = fp.NoseTip[1]
                
                # Get two half images
                if fp.FaceOrientation == 'right':
                    image_halves = [vidFrame[:midWidth, :], vidFrame[midWidth:, :]]
                    image_side = ['Top', 'Bottom']
                elif fp.FaceOrientation == 'left':
                    image_halves = [vidFrame[midWidth:, :], vidFrame[:midWidth, :]]
                    image_side = ['Bottom', 'Top']

            # By convention, always start with the left side of the face
            face_side = ['Left', 'Right']

        else:
            image_halves = np.array([vidFrame])
            image_side = ['Full']
            face_side = ['Unknown']

        # Get whisking parameters for each side, and save them to the WhiskingParams object
        # initialize values
        wp = [WhiskingParams(None, None, None), WhiskingParams(None, None, None)]
        for image, f_side, im_side, side_id in zip(image_halves, face_side, image_side, range(len(image_halves))):
            wp[side_id] = WhiskingFun.find_whiskerpad(image, fp, face_side=f_side, image_side=im_side)           
            wp[side_id].ImageSide = im_side
            wp[side_id].ImageDimensions = image.shape
            wp[side_id].ImageBorderAxis, wp[side_id].ProtractionDirection, wp[side_id].LinkingDirection = WhiskingFun.get_whisking_params(wp[side_id])

        return wp, splitUp

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
            # whiskingParams = WhiskingFun.get_whisking_params(vidFrame,interactive=True)
            wpCoordinates, wpLocation, wpRelativeLocation = WhiskingFun.find_whiskerpad_interactive(vidFrame)
            # Clear variables firstVideo
        elif splitUp.lower() == "yes":
            whiskparams, initialFrame = WhiskingFun.get_nose_tip_coordinates(vid.Path, vid.Name)
            midWidth = round(vidFrame.shape[1] / 2)
            if midWidth - vidFrame.shape[1] / 8 < fp.NoseTip[0] < midWidth + vidFrame.shape[1] / 8:
                # If nosetip x value within +/- 1/8 of frame width, use that value
                midWidth = fp.NoseTip[0]
            
            # Get whisking parameters for left side
            leftImage = vidFrame[:, :midWidth]
            # whiskingParams = WhiskingFun.get_whisking_params(leftImage, midWidth - round(vidFrame.shape[1] / 2), fp.NoseTip, fp.FaceAxis=None, face_orientation=None, image_side='Left', interactive=True)
            wpCoordinates, wpLocation, wpRelativeLocation = WhiskingFun.find_whiskerpad_interactive(leftImage, midWidth - round(vidFrame.shape[1] / 2), fp.NoseTip, face_axis=None, face_orientation=None, image_side='Left')
            
            # Save values to whiskingParams 
            TBD
            # Get whisking parameters for right side
            rightImage = vidFrame[:, midWidth:]
            # whiskingParams[1] = WhiskingFun.get_whisking_params(rightImage, round(vidFrame.shape[1] / 2) - midWidth, fp.NoseTip, face_axis=None, face_orientation=None, image_side='Right', interactive=True)
            wpCoordinates, wpLocation, wpRelativeLocation = WhiskingFun.find_whiskerpad_interactive(rightImage, midWidth - round(vidFrame.shape[1] / 2), fp.NoseTip, face_axis=None, face_orientation=None, image_side='Right')
            
            whiskingParams[0].ImageSide = 'Left'
            whiskingParams[1].ImageSide = 'Right'

        return whiskingParams, splitUp
            
    @staticmethod
    def find_whiskerpad(topviewImage, fp, face_side, image_side):
        
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

        contour_brightSide = WhiskingFun.get_side_brightness(opening, contour)

        # At this point, the contour is roughly a right triangle: 
        # two straight sides are the image border, and the face contour is the "hypothenuse".
        # Extract the face outline from this contour.

        if fp.FaceAxis == 'vertical':
            if fp.FaceOrientation == 'down':
                if contour_brightSide['maxSide'] == 'right':
                    # starting point is the point with the lowest x value and lowest y value
                    starting_point = contour[contour[:, 0, 1].argmin(), 0, :]
                    # ending point is the point with the highest x value and highest y value
                    ending_point = contour[contour[:, 0, 1].argmax(), 0, :]
                if contour_brightSide['maxSide'] == 'left':
                    # starting point is the point with the lowest x value and highest y value
                    starting_point = contour[contour[:, 0, 1].argmax(), 0, :]
                    # ending point is the point with the lowest x value and highest y value
                    ending_point = contour[contour[:, 0, 0].argmax(), 0, :]
            elif fp.FaceOrientation == 'up':
                if contour_brightSide['maxSide'] == 'right':
                    # starting point is the point with the highest x value and lowest y value
                    starting_point = contour[contour[:, 0, 1].argmax(), 0, :]
                    # ending point is the point with the lowest x value and lowest y value
                    ending_point = contour[contour[:, 0, 1].argmin(), 0, :]
                if contour_brightSide['maxSide'] == 'left':
                    # starting point is the point with the highest x value and highest y value
                    starting_point = contour[contour[:, 0, 1].argmin(), 0, :]
                    # ending point is the point with the lowest x value and highest y value
                    ending_point = contour[contour[:, 0, 0].argmax(), 0, :]
        elif fp.FaceAxis == 'horizontal':
            if fp.FaceOrientation == 'left':
                if contour_brightSide['maxSide'] == 'top':
                    # starting point is the point with the lowest x value and lowest y value
                    starting_point = contour[contour[:, 0, 0].argmin(), 0, :]
                    # ending point is the point with the highest x value and lowest y value
                    ending_point = contour[contour[:, 0, 0].argmax(), 0, :]
                if contour_brightSide['maxSide'] == 'bottom':
                    # contour = np.fliplr(contour)
                    # contour = np.flipud(contour)
                    # starting point is the point with the highest x value and lowest y value
                    starting_point = contour[contour[:, 0, 1].argmin(), 0, :]
                    # ending point is the point with the lowest x value and lowest y value
                    ending_point = contour[contour[:, 0, 1].argmax(), 0, :]
            elif fp.FaceOrientation == 'right':
                if contour_brightSide['maxSide'] == 'top':
                    contour = np.flipud(contour)
                    # starting point is the point with the lowest x value and highest y value
                    starting_point = contour[contour[:, 0, 1].argmax(), 0, :]
                    # ending point is the point with the highest x value and lowest y value
                    ending_point = contour[contour[:, 0, 0].argmax(), 0, :]
                if contour_brightSide['maxSide'] == 'bottom':
                    # starting point is the point with the lowest x value and lowest y value
                    starting_point = contour[contour[:, 0, 1].argmin(), 0, :]
                    # ending point is the point with the highest x value and highest y value
                    ending_point = contour[contour[:, 0, 0].argmax(), 0, :]

        # Find the index of the starting point in the contour
        starting_point_index = np.where((contour == starting_point).all(axis=2))[0][0]
        # Find the index of the ending point in the contour
        ending_point_index = np.where((contour == ending_point).all(axis=2))[0][0]
 
        if fp.FaceOrientation == 'up' or (fp.FaceOrientation == 'right' and contour_brightSide['maxSide'] == 'bottom'):
            # add modulus contour length to starting point index
            starting_point_index = starting_point_index + contour.shape[0]

        # Make sure starting point index is smaller than ending point index
        if starting_point_index > ending_point_index:
            starting_point_index, ending_point_index = ending_point_index, starting_point_index

        # face contour is the part of the contour bounded by those indices
        face_contour = contour[starting_point_index:ending_point_index+1, 0, :]

        # Plot rotated contour face_contour_r
        fig, ax = plt.subplots()
        ax.plot(face_contour[:, 0], face_contour[:, 1], linewidth=2, color='r')

        # Assuming a straight line between the starting point and the ending point,
        # we rotate the curve to set that line as the x-axis. We then find the highest y value 
        # on the rotated contour, and use that as the index for the whisker pad location.

        # Find the angle of the straight line
        theta = np.arctan((ending_point[1] - starting_point[1]) / (ending_point[0] - starting_point[0]))

        if (fp.FaceOrientation == 'right' and contour_brightSide['maxSide'] == 'top') \
            or (fp.FaceOrientation == 'left' and contour_brightSide['maxSide'] == 'bottom'):
            # Rotate the contour by that angle minus pi, in the counter clockwise direction
            rot = np.array([[np.cos(theta - np.pi), -np.sin(theta - np.pi)], [np.sin(theta - np.pi), np.cos(theta - np.pi)]])
        else:
            # Rotate the contour by angle, in the clockwise direction
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        if fp.FaceOrientation == 'down' or fp.FaceOrientation == 'left':
            face_contour_r = np.round(np.dot(face_contour - starting_point, rot) + starting_point)
        else:
            face_contour_r = np.round(np.dot(starting_point - face_contour, rot) + starting_point)

        # Plot rotated contour face_contour_r
        fig, ax = plt.subplots()
        ax.plot(face_contour_r[:, 0], face_contour_r[:, 1], linewidth=2, color='r')

        # Find the index of the highest y value on the rotated contour
        wpLocationIndex = face_contour_r[:, 1].argmax()

        # The whisker pad location in the original image is the whisker pad location index of the face contour 
        wpLocation = face_contour[wpLocationIndex, :]

        # # Plot image and wpLocation on top
        # fig, ax = plt.subplots()
        # ax.imshow(topviewImage)
        # plt.title('Face contour')
        # ax.plot(face_contour[:, 0], face_contour[:, 1], linewidth=2, color='r')
        # ax.plot(wpLocation[0], wpLocation[1], 'o', color='y')
        # plt.show()

        # Add create fp object for that whiskerpad and face side
        fp_wp = fp.copy()
        fp_wp.FaceSide = face_side

        # if is within 1/3 of the image width from the nose tip (or midline), in the orthogonal direction of the head axis, keep it
        keep_wp_location = False
        if fp_wp.FaceAxis == 'vertical':
                # Adjust nose tip coordinates if needed
                if (fp_wp.FaceOrientation == 'down' and face_side == 'Left') or (fp_wp.FaceOrientation == 'up' and face_side == 'Right'):
                    fp_wp.NoseTip[0] = 0
                else:
                    fp_wp.NoseTip[0] = topviewImage.shape[1]
                keep_wp_location = np.abs(wpLocation[0] - fp_wp.NoseTip[0]) < topviewImage.shape[1] / 3

        elif fp_wp.FaceAxis == 'horizontal':
                # Adjust nose tip coordinates if needed
                if (fp_wp.FaceOrientation == 'left' and face_side == 'Left') or (fp_wp.FaceOrientation == 'right' and face_side == 'Right'):
                    fp_wp.NoseTip[1] = 0
                else:
                    fp_wp.NoseTip[1] = topviewImage.shape[0]
                keep_wp_location = np.abs(wpLocation[1] - fp_wp.NoseTip[1]) < topviewImage.shape[0] / 3
                
        if keep_wp_location:
            # Save the image with the contour overlayed and the whisker pad location labelled on it
            image_with_contour = topviewImage.copy()
            cv2.drawContours(image_with_contour, [face_contour], -1, (0, 255, 0), 3)
            # define file name based on face orientation 
            output_path = os.path.join(video_dir, 'whiskerpad_' + image_side.lower() + '.jpg')
            cv2.imwrite(output_path, cv2.circle(image_with_contour, tuple(wpLocation), 10, (255, 0, 0), -1))

            # Define whisker pad area (wpArea) as the rectangle around the whisker pad location
            wpArea = [wpLocation[0] - 10, wpLocation[1] - 10, 20, 20]

        else:
            # Get ballpark whisker pad coordinates, according to nose tip, face_axis and face orientation
            # Define whiskerpad area (wpArea) as the rectangle around the nose tip, offset by some ratio towards the face
            if fp_wp.FaceAxis == 'vertical':
                wp_offset = [topviewImage.shape[1] / 8, topviewImage.shape[0] / 4]
                if fp_wp.FaceOrientation == 'up':
                    wpArea = [fp_wp.NoseTip[0] - wp_offset[0], fp_wp.NoseTip[1] + wp_offset[1], wp_offset[0], wp_offset[1]]
                elif fp_wp.FaceOrientation == 'down':
                    wpArea = [fp_wp.NoseTip[0] - wp_offset[0], fp_wp.NoseTip[1] - wp_offset[1], wp_offset[0], wp_offset[1]]
            elif fp_wp.FaceAxis == 'horizontal':
                wp_offset = [topviewImage.shape[1] / 8, topviewImage.shape[0] / 4]
                if fp_wp.FaceOrientation == 'left':
                    wpArea = [fp_wp.NoseTip[0] + wp_offset[0], fp_wp.NoseTip[1] - wp_offset[1], wp_offset[0], wp_offset[1]]
                elif fp_wp.FaceOrientation == 'right':
                    wpArea = [fp_wp.NoseTip[0] - wp_offset[0], fp_wp.NoseTip[1] - wp_offset[1], wp_offset[0], wp_offset[1]]

            # Define whisker pad location (wpLocation) as the center of the whisker pad area
            wpLocation = np.round([wpArea[0] + wpArea[2] / 2, wpArea[1] + wpArea[3] / 2])

        # Define whisker pad relative location (wpRelativeLocation) as the whisker pad location divided by the image dimensions
        wpRelativeLocation = [wpLocation[0] / topviewImage.shape[1], wpLocation[1] / topviewImage.shape[0]]

        # plot image, and overlay whisker pad area contour and location on top
        # fig, ax = plt.subplots()
        # ax.imshow(topviewImage)
        # plt.title('Draw rectangle around whisker pad')
        # wpAttributes = plt.Rectangle((wpArea[0], wpArea[1]), wpArea[2], wpArea[3], label='align', visible=False, fill=False)
        # ax.add_patch(wpAttributes)
        # plt.show()

        whiskingParams = WhiskingParams(wpArea, wpLocation, wpRelativeLocation, fp_wp)

        return whiskingParams

    @staticmethod
    def get_side_brightness(wpImage, contour=None):

        # If contour is provided, crop image to rectangle defined by contour extreme points
        if contour is not None:
            extrema = contour[:, 0, :]
            wpImage = wpImage[extrema[:, 1].min():extrema[:, 1].max(), extrema[:, 0].min():extrema[:, 0].max()]

        # Find brightness for each side
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

        # Identify the side with the highest brightness (remember that the image is thresholded with an inverse binary threshold)
        sideBrightness['maxSide']=max(sideBrightness, key=sideBrightness.get) 
    
        # Add side brightness ratio
        sideBrightness['top_bottom_ratio'] = top_brightness / bottom_brightness
        sideBrightness['left_right_ratio'] = left_brightness / right_brightness

        return sideBrightness

    @staticmethod
    def find_whiskerpad_interactive(topviewImage):
        fig, ax = plt.subplots()
        ax.imshow(topviewImage)
        plt.title('Draw rectangle around whisker pad')
        wpAttributes = plt.Rectangle((0, 0), 1, 1, label='align', visible=False, fill=False)
        ax.add_patch(wpAttributes)
        plt.show()

        wpArea = wpAttributes.get_bbox().bounds
        wpLocation = np.round([wpArea[0] + wpArea[2] / 2, wpArea[1] + wpArea[3] / 2])
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
    def get_whisking_params(wp):

        if wp.FaceAxis == 'horizontal':
            ImageBorderAxis = 'bottom' if wp.RelativeLocation[1] > 0.5 else 'top'
            protractionDirection = 'leftward' if wp.NoseTip[0] < wp.Location[0] else 'rightward'
        else:
            ImageBorderAxis = 'right' if wp.RelativeLocation[0] > 0.5 else 'left'
            protractionDirection = 'upward' if wp.NoseTip[1] < wp.Location[1] else 'downward'

        linkingDirection = 'rostral'

        return ImageBorderAxis, protractionDirection, linkingDirection

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
        # initialFrame = -1
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

        # Get the current frame number
        initialFrame=vidcap.get(cv2.CAP_PROP_POS_FRAMES)-1

        # Close the video file
        vidcap.release()

        sideBrightness=WhiskingFun.get_side_brightness(opening, contour)

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
        nose_tip = nose_tip + np.array([image.shape[1]//2-200, image.shape[0]//2-200])

        # Find the midline offset
        midline_offset = np.abs(nose_tip[0] - image.shape[1] / 2)

        # Save the frame with the nose tip labelled on it
        cv2.imwrite(os.path.join(video_dir, 'nose_tip.jpg'), cv2.circle(image, tuple(nose_tip), 10, (255, 0, 0), -1))

        # instanciate whiskparams with nose_tip, face_axis, face_orientation
        face_params = FaceParams(face_axis, face_orientation, None , nose_tip, midline_offset)

        return face_params, initialFrame

    @staticmethod
    def save_whiskerpad_params(whiskingParams, videofile):
        trackingDir = os.path.dirname(videofile)
        filename = os.path.basename(videofile).split('.')[0]
        # Save whiskerpad parameters to json file named whiskerpad_{filename}.json
        with open(os.path.join(trackingDir, 'whiskerpad_' + filename + '.json'), 'w') as file:
        # with open(os.path.join(trackingDir, 'whiskerpad.json'), 'w') as file:
            json.dump(whiskingParams, file, indent='\t', cls=WhiskingParamsEncoder)

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
        whiskerpadParams, splitUp = WhiskingFun.draw_whiskerpad_roi(args.videofile, args.splitUp)
    else:
        whiskerpadParams, splitUp = WhiskingFun.get_whiskerpad_params(args.videofile, args.splitUp)

    # Create whiskerpadParams dictionary
    whiskerpad = {'filename': os.path.basename(args.videofile),
                        'directory': os.path.dirname(args.videofile), 
                        'split': splitUp,
                        'whiskerpads': whiskerpadParams}

    # Save whisking parameters to json file
    WhiskingFun.save_whiskerpad_params(whiskerpad, args.videofile)

    print(whiskerpad)
