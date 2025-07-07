"""whiskerpad.py
Methods for extracting whisker pad coordinates from video files

Part of the Whisker Tracking pipeline for analyzing rodent whisker data.

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

class WhiskingParams:
    """
    Class for storing whisking parameters for a whisker pad
    """
    def __init__(self, wpArea, wpLocation, wpRelativeLocation, fp=None):
        """
        Initialize the whisking parameters object

        Parameters
        ----------
        wpArea : list
            List of 4 integers representing the whisker pad area coordinates (x, y, width, height)
        wpLocation : list
            List of 2 integers representing the whisker pad location (x, y)
        wpRelativeLocation : list
            List of 2 floats representing the whisker pad relative location (x, y)
        fp : FaceParams
            Face parameters object containing the face axis, face orientation, face side, nose tip and midline offset

        The whisking parameters object contains the following attributes:
        FaceAxis : str. The axis of the face (horizontal or vertical)
        FaceOrientation : str. The orientation of the face (up, down, left, right)
        FaceSide : str. The side of the face (left, right)
        NoseTip : list. The coordinates of the nose tip (x, y)
        MidlineOffset : int. The offset of the midline from the center of the image
        AreaCoordinates : list. The coordinates of the whisker pad area (x, y, width, height)
        Location : list. The coordinates of the whisker pad location (x, y)
        RelativeLocation : list. The relative coordinates of the whisker pad location (x, y) w/r to the image dimensions
        ImageSide : str. The side of the image (left, right, top, bottom)
        ImageBorderAxis : str. The image border corresponding to the long axis of the face, if any (top, bottom, left, right) 
        ProtractionDirection : str. The protraction direction of the whisker pad (upward, downward, leftward, rightward)
        LinkingDirection : str. The linking direction of the whisker pad (rostral, caudal)
        ImageCoordinates : list. The coordinates of the image if cropped (x, y, width, height)
        """

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
        self.ImageCoordinates = []

class WhiskingParamsEncoder(json.JSONEncoder):
    """
    JSON encoder for WhiskingParams objects
    """
    def default(self, obj):
        """
        Convert WhiskingParams object to a dictionary for JSON serialization
        """
        if isinstance(obj, WhiskingParams):
            # Convert WhiskingParams object to a dictionary
            params_dict = obj.__dict__.copy()
            # Convert ndarray objects to lists
            if isinstance(params_dict['Location'], np.ndarray):
                params_dict['Location'] = params_dict['Location'].tolist()
            if isinstance(params_dict['NoseTip'], np.ndarray):
                params_dict['NoseTip'] = params_dict['NoseTip'].tolist()
            # Convert numpy types to native Python types
            if isinstance(params_dict['MidlineOffset'], (np.int64, np.float64)):
                params_dict['MidlineOffset'] = float(params_dict['MidlineOffset'])
            # Convert AreaCoordinates numpy types to native Python types
            if isinstance(params_dict['AreaCoordinates'], list):
                params_dict['AreaCoordinates'] = [int(x) if isinstance(x, (np.int32, np.int64)) else x for x in params_dict['AreaCoordinates']]
            # Convert RelativeLocation numpy types to native Python types
            if isinstance(params_dict['RelativeLocation'], list):
                params_dict['RelativeLocation'] = [float(x) if isinstance(x, (np.float64, np.float32)) else x for x in params_dict['RelativeLocation']]
            # Convert ImageCoordinates numpy types to native Python types
            if isinstance(params_dict['ImageCoordinates'], tuple):
                params_dict['ImageCoordinates'] = tuple(int(x) if isinstance(x, (np.int32, np.int64)) else x for x in params_dict['ImageCoordinates'])
            # Return the updated dictionary
            return params_dict
        # Handle numpy types directly
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class FaceParams:
    """
    Class for storing face parameters
    """
    def __init__(self, face_axis, face_orientation, face_side=None, nose_tip=None, midline_offset=0):
        """
        Initialize the face parameters object

        Parameters
        ----------
        face_axis : str. The axis of the face (horizontal or vertical)
        face_orientation : str. The orientation of the face (up, down, left, right)
        face_side : str. The side of the face (left, right)
        nose_tip : list. The coordinates of the nose tip (x, y)
        midline_offset : int. The offset of the midline from the center of the image

        The face parameters object contains the following attributes:
        FaceAxis : str. The axis of the face (horizontal or vertical)
        FaceOrientation : str. The orientation of the face (up, down, left, right)
        FaceSide : str. The side of the face (left, right)
        NoseTip : list. The coordinates of the nose tip (x, y)
        MidlineOffset : int. The offset of the midline from the center of the image
        """

        self.FaceAxis = face_axis
        self.FaceOrientation = face_orientation
        self.FaceSide = face_side
        self.NoseTip = nose_tip
        self.MidlineOffset = midline_offset
    
    def copy(self):
        return copy.deepcopy(self)
    
class Params:
    """
    Class for storing parameters for whisker pad extraction
    """
    def __init__(self, video_file, splitUp=False, basename=None, interactive=False):
        """
        Initialize the parameters object

        Parameters
        ----------
        video_file : str. The path to the video file
        splitUp : bool. Whether to split the video into left and right sides
        basename : str. The basename of the video file
        interactive : bool. Whether to run in interactive mode

        The parameters object contains the following attributes:
        video_file : str. The path to the video file
        basename : str. The basename of the video file
        video_dir : str. The directory containing the video file
        splitUp : bool. Whether to split the video into left and right sides
        interactive : bool. Whether to run in interactive mode
        """

        self.video_file = video_file
        if basename is None:
            self.basename = os.path.splitext(os.path.basename(video_file))[0]
        else:
            self.basename = basename
        self.video_dir = os.path.dirname(video_file)
        self.splitUp = splitUp
        self.interactive = interactive

class WhiskerPad:
    """
    Class for extracting whisker pad coordinates from video files
    """
    @staticmethod
    def get_whiskerpad_params(args):
        """
        Get whisker pad parameters from video file

        Parameters
        ----------
        args : Params. The parameters object containing the video file, splitUp flag, and basename

        Returns
        -------
        whiskingParams : list. A list of WhiskingParams objects containing the whisker pad parameters
        splitUp : bool. Whether to split the video into left and right sides
        """

        # Get the face sides
        image_halves, image_side, face_side, fp = get_side_image(args.video_file, args.splitUp, args.video_dir)

        # Get whisking parameters for each side, and save them to the WhiskingParams object
        # initialize values
        wp = [WhiskingParams(None, None, None), WhiskingParams(None, None, None)]
        for image, f_side, im_side, side_id in zip(image_halves, face_side, image_side, range(len(image_halves))):
            wp[side_id] = WhiskerPad.find_whiskerpad(image, fp, face_side=f_side, image_side=im_side, video_dir=args.video_dir, basename=args.basename)    
            
            # Check if whiskerpad detection was successful
            if wp[side_id] is None:
                print(f"Warning: Failed to detect whiskerpad for {f_side} side. Creating default parameters.")
                # Create default whiskerpad parameters
                wp[side_id] = WhiskerPad.create_default_whiskerpad(image, fp, face_side=f_side, image_side=im_side)
            
            wp[side_id].ImageSide = im_side
            # Set image coordinates as a tuple of image coordinates x, y, width, height
            if im_side == 'left' or im_side == 'top':
                wp[side_id].ImageCoordinates = tuple([0, 0, image.shape[1], image.shape[0]])
            elif im_side == 'right':
                    wp[side_id].ImageCoordinates = tuple([fp.NoseTip[0], 0, image.shape[1], image.shape[0]])
            elif im_side == 'bottom':
                    wp[side_id].ImageCoordinates = tuple([0, fp.NoseTip[1], image.shape[1], image.shape[0]])
            wp[side_id].ImageBorderAxis, wp[side_id].ProtractionDirection, wp[side_id].LinkingDirection = WhiskerPad.get_whisking_params(wp[side_id])

        return wp, args.splitUp

    @staticmethod
    def draw_whiskerpad_roi(args):
        vidFrame = cv2.cvtColor(vid.read()[1], cv2.COLOR_BGR2GRAY)
        cv2.imshow("Video Frame", vidFrame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if args.splitUp is None:
            args.splitUp = input("Do you need to split the video? (Yes/No): ")

        if not args.splitUp:
            # Get whisker pad coordinates
            # Get whisking parameters
            # whiskingParams = WhiskerPad.get_whisking_params(vidFrame,interactive=True)
            wpCoordinates, wpLocation, wpRelativeLocation = WhiskerPad.find_whiskerpad_interactive(vidFrame)
            # Clear variables firstVideo
        elif args.splitUp:
            whiskparams, initialFrame = WhiskerPad.get_nose_tip_coordinates(vid.Path, vid.Name)
            midWidth = round(vidFrame.shape[1] / 2)
            if midWidth - vidFrame.shape[1] / 8 < fp.NoseTip[0] < midWidth + vidFrame.shape[1] / 8:
                # If nosetip x value within +/- 1/8 of frame width, use that value
                midWidth = fp.NoseTip[0]
            
            # Get whisking parameters for left side
            leftImage = vidFrame[:, :midWidth]
            # whiskingParams = WhiskerPad.get_whisking_params(leftImage, midWidth - round(vidFrame.shape[1] / 2), fp.NoseTip, fp.FaceAxis=None, face_orientation=None, image_side='left', interactive=True)
            wpCoordinates, wpLocation, wpRelativeLocation = WhiskerPad.find_whiskerpad_interactive(leftImage, midWidth - round(vidFrame.shape[1] / 2), fp.NoseTip, face_axis=None, face_orientation=None, image_side='left')
            
            # Save values to whiskingParams 
            TBD
            # Get whisking parameters for right side
            rightImage = vidFrame[:, midWidth:]
            # whiskingParams[1] = WhiskerPad.get_whisking_params(rightImage, round(vidFrame.shape[1] / 2) - midWidth, fp.NoseTip, face_axis=None, face_orientation=None, image_side='right', interactive=True)
            wpCoordinates, wpLocation, wpRelativeLocation = WhiskerPad.find_whiskerpad_interactive(rightImage, midWidth - round(vidFrame.shape[1] / 2), fp.NoseTip, face_axis=None, face_orientation=None, image_side='right')
            
            whiskingParams[0].ImageSide = 'left'
            whiskingParams[1].ImageSide = 'right'

        return whiskingParams, args.splitUp
    
    @staticmethod
    def distance_to_contour(contour, feature_coord):
        """
        Calculate the Euclidean distance between the centroid of a contour and a feature coordinate (e.g. NoseTip)
        
        Arguments:
        contour -- the contour
        feature_coord -- the feature coordinate
        
        Returns:
        distance -- the Euclidean distance between the centroid of the contour and the feature coordinate       
        """
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Calculate Euclidean distance between centroid and the feature coordinate
        centroid = np.array([cx, cy])
        distance = np.linalg.norm(centroid - feature_coord)

        return distance
            
    @staticmethod
    def find_whiskerpad(topviewImage, fp, face_side, image_side, video_dir=None, basename=None):
        """
        Find the whisker pad location in the topview image
        
        Arguments:
        topviewImage -- the topview image
        fp -- face parameters object
        face_side -- the side of the face (left, right)
        image_side -- the side of the image (left, right, top, bottom)
        video_dir -- the directory containing the video file
        basename -- the basename of the video file (for saving plots)
        
        Returns:
        whiskingParams -- the whisking parameters object
        """
                    
        # Threshold the first frame and apply morphological opening to the binary image
        _, opening = WhiskerPad.morph_open(topviewImage)

        # Find contours in the opened morphology
        contours = WhiskerPad.find_contours(opening)

        # Filter contours based on minimum area threshold
        minArea = 0.1 * (topviewImage.shape[0] * topviewImage.shape[1])
        filteredContours = [cnt for cnt in contours if cv2.contourArea(cnt) >= minArea]

        # If two or more contours are found, keep the one closest to the NoseTip
        if len(filteredContours) > 1:
            contour = min(filteredContours, key=lambda cnt: WhiskerPad.distance_to_contour(cnt, fp.NoseTip))
        elif filteredContours:
            contour = filteredContours[0]  # If only one valid contour, just use it
        else:
            contour = None
            return None

        # # plot image, and overlay the contour on top
        # fig, ax = plt.subplots()
        # ax.imshow(topviewImage)
        # plt.title('Face contour')
        # ax.plot(contour[:, 0, 0], contour[:, 0, 1], linewidth=2, color='r')
        # plt.show()

        contour_brightSide = WhiskerPad.get_side_brightness(opening, contour)

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

        try:
            # Find the index of the starting point in the contour
            starting_point_index = np.where((contour == starting_point).all(axis=2))[0][0]
            # Find the index of the ending point in the contour
            ending_point_index = np.where((contour == ending_point).all(axis=2))[0][0]
        except IndexError:
            raise ValueError("Starting or ending point not found in the contour")
 
        # Adjust starting point index based on face orientation
        if fp.FaceOrientation == 'up' or (fp.FaceOrientation == 'right' and contour_brightSide['maxSide'] == 'bottom'):
            # Add modulus contour length to starting point index
            starting_point_index = (starting_point_index + contour.shape[0]) % contour.shape[0]

        # Ensure starting point index is smaller than ending point index
        if starting_point_index > ending_point_index:
            starting_point_index, ending_point_index = ending_point_index, starting_point_index

        # Extract the face contour bounded by the starting and ending points
        face_contour = contour[starting_point_index:ending_point_index+1, 0, :]

        # Plot rotated contour face_contour_r
        # fig, ax = plt.subplots()
        # ax.plot(face_contour[:, 0], face_contour[:, 1], linewidth=2, color='r')

        # Assuming a straight line between the starting point and the ending point,
        # we rotate the curve to set that line as the x-axis. We then find the highest y value 
        # on the rotated contour, and use that as the index for the whisker pad location.

        # Calculate the angle of the straight line
        if ending_point[0] != starting_point[0]:
            theta = np.arctan((ending_point[1] - starting_point[1]) / (ending_point[0] - starting_point[0]))
        else:
            # Handle division by zero by setting theta to 90 degrees
            theta = np.pi / 2
            
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
        # fig, ax = plt.subplots()
        # ax.plot(face_contour_r[:, 0], face_contour_r[:, 1], linewidth=2, color='r')

        # Find the index of the highest y value on the rotated contour
        wpLocationIndex = face_contour_r[:, 1].argmax()

        # The whisker pad location in the original image is the whisker pad location index of the face contour 
        wpLocation = face_contour[wpLocationIndex, :]

        # # Plot image and wpLocation on top
        # fig, ax = plt.subplots()
        # ax.imshow(topviewImage)
        # plt.title(f"Face contour with whisker pad location for face side ({face_side})")
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
                if (fp_wp.FaceOrientation == 'down' and face_side == 'left') or (fp_wp.FaceOrientation == 'up' and face_side == 'right'):
                    fp_wp.NoseTip[0] = 0
                else:
                    fp_wp.NoseTip[0] = topviewImage.shape[1]
                keep_wp_location = np.abs(wpLocation[0] - fp_wp.NoseTip[0]) < topviewImage.shape[1] / 3

        elif fp_wp.FaceAxis == 'horizontal':
                # Adjust nose tip coordinates if needed
                if (fp_wp.FaceOrientation == 'left' and face_side == 'left') or (fp_wp.FaceOrientation == 'right' and face_side == 'right'):
                    fp_wp.NoseTip[1] = 0
                else:
                    fp_wp.NoseTip[1] = topviewImage.shape[0]
                keep_wp_location = np.abs(wpLocation[1] - fp_wp.NoseTip[1]) < topviewImage.shape[0] / 3
                
        if keep_wp_location:
            if video_dir is not None:
                # Save the image with the contour overlayed and the whisker pad location labelled on it
                image_with_contour = topviewImage.copy()
                cv2.drawContours(image_with_contour, [face_contour], -1, (0, 255, 0), 3)
                # Save image with whisker pad location labelled
                plot_dir = os.path.join(video_dir, 'plots')
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir, exist_ok=True)
                # define file name based on face orientation 
                output_path = os.path.join(plot_dir, f'whiskerpad_{basename}_{face_side.lower()}.jpg')
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
    def get_nose_tip_coordinates(videoFileName, video_dir=None):
        # Check that the video file exists
        if not os.path.exists(videoFileName):
            print("Error: Video file does not exist. Check if there is a typo in the file name or if the file is in the correct directory.")
            print("File name: ", videoFileName)
            print("Directory file list: ", os.listdir(os.path.dirname(videoFileName)))
            return None, None
        
        # Open the video file 
        vidcap = cv2.VideoCapture(videoFileName)

        # Check that video file is open
        if not vidcap.isOpened():
            print("Error: Could not open video file")
            return None, None
        
        contour=None
        # initialFrame = -1
        while contour is None or len(contour) == 0:
            # open the next frame
            _, image = vidcap.read()

            # Threshold the first frame and apply morphological opening to the binary image
            gray, opening = WhiskerPad.morph_open(image, crop=True)

            # Find contours in the opened morphology
            contours = WhiskerPad.find_contours(opening)

            # Filter contours based on minimum area threshold
            minArea = 6000
            filteredContours = [cnt for cnt in contours if cv2.contourArea(cnt) >= minArea]

            # Find the largest contour
            contour = max(filteredContours, key=cv2.contourArea)

        # Get the current frame number
        initialFrame=vidcap.get(cv2.CAP_PROP_POS_FRAMES)-1

        # Close the video file
        vidcap.release()

        sideBrightness=WhiskerPad.get_side_brightness(opening, contour)

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

        if video_dir is not None:
            plot_dir = os.path.join(video_dir, 'plots')
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir, exist_ok=True)
            image_path = os.path.join(plot_dir, f'{os.path.splitext(os.path.basename(videoFileName))[0]}_nose_tip.jpg')
            print(f"Saving image with nose tip labelled at {image_path}")
            # Save the frame with the nose tip labelled on it
            cv2.imwrite(image_path, cv2.circle(image, tuple(nose_tip), 10, (255, 0, 0), -1))

        # instanciate whiskparams with nose_tip, face_axis, face_orientation
        face_params = FaceParams(face_axis, face_orientation, None , nose_tip, midline_offset)

        return face_params, initialFrame

    @staticmethod
    def save_whiskerpad_params(args, whiskerpadParams):
        
        trackingDir = args.video_dir

        # Create whiskerpadParams dictionary
        whiskerpad = {'filename': os.path.basename(args.video_file),
                        'basename': args.basename,
                        'directory': trackingDir,
                        'split': args.splitUp,
                        'whiskerpads': whiskerpadParams}

        filename = whiskerpad['basename']
        #os.path.basename(args.video_file).split('.')[0]

        # Save whiskerpad parameters to json file
        with open(os.path.join(trackingDir, 'whiskerpad_' + filename + '.json'), 'w') as file:
            json.dump(whiskerpad, file, indent='\t', cls=WhiskingParamsEncoder)

        return whiskerpad

    @staticmethod
    def create_default_whiskerpad(image, fp, face_side, image_side):
        """
        Create default whiskerpad parameters when automatic detection fails.
        
        This method creates reasonable default parameters based on the image dimensions
        and face parameters, allowing the pipeline to continue even when contour detection fails.
        
        Parameters
        ----------
        image : numpy.ndarray
            The image for which to create default whiskerpad parameters
        fp : FaceParams
            Face parameters object containing face orientation and nose tip
        face_side : str
            The side of the face ('left' or 'right')
        image_side : str
            The side of the image ('left', 'right', 'top', 'bottom')
            
        Returns
        -------
        WhiskingParams
            A WhiskingParams object with default parameters
        """
        print(f"Creating default whiskerpad parameters for {face_side} side")
        
        # Create a copy of face parameters for this whiskerpad
        fp_wp = fp.copy()
        fp_wp.FaceSide = face_side
        
        # Define default whiskerpad area based on image dimensions and face orientation
        img_height, img_width = image.shape[:2]
        
        if fp.FaceAxis == 'vertical':
            # For vertical face axis, whiskerpad is typically on the side
            wp_width = img_width // 6  # Use 1/6 of image width
            wp_height = img_height // 4  # Use 1/4 of image height
            
            if face_side == 'left':
                # Left whiskerpad is on the left side of the image
                wp_x = wp_width // 2
            else:
                # Right whiskerpad is on the right side of the image
                wp_x = img_width - wp_width - (wp_width // 2)
                
            # Vertical center, slightly offset based on face orientation
            if fp.FaceOrientation == 'up':
                wp_y = img_height // 3  # Upper third
            else:
                wp_y = (2 * img_height) // 3  # Lower third
                
        else:  # horizontal face axis
            # For horizontal face axis, whiskerpad is typically on top/bottom
            wp_width = img_width // 4  # Use 1/4 of image width
            wp_height = img_height // 6  # Use 1/6 of image height
            
            if face_side == 'left':
                # Left whiskerpad position
                wp_x = img_width // 3
            else:
                # Right whiskerpad position
                wp_x = (2 * img_width) // 3
                
            if fp.FaceOrientation == 'left':
                wp_y = wp_height // 2
            else:
                wp_y = img_height - wp_height - (wp_height // 2)
        
        # Define whiskerpad area [x, y, width, height]
        wpArea = [wp_x - wp_width//2, wp_y - wp_height//2, wp_width, wp_height]
        
        # Ensure coordinates are within image bounds
        wpArea[0] = max(0, min(wpArea[0], img_width - wpArea[2]))
        wpArea[1] = max(0, min(wpArea[1], img_height - wpArea[3]))
        
        # Define whiskerpad location as center of area
        wpLocation = np.array([wpArea[0] + wpArea[2] // 2, wpArea[1] + wpArea[3] // 2])
        
        # Define relative location
        wpRelativeLocation = [wpLocation[0] / img_width, wpLocation[1] / img_height]
        
        # Create WhiskingParams object
        whiskingParams = WhiskingParams(wpArea, wpLocation, wpRelativeLocation, fp_wp)
        
        return whiskingParams

    # ...existing code...
def get_side_image(video_file, splitUp, video_dir=None):
    """
    Get the left and right side images of the face

    Parameters
    ----------
    video_file : str. The path to the video file
    splitUp : bool. Whether to split the video into left and right sides
    video_dir : str. The directory containing the video file

    Returns
    -------
    image_halves : list. A list of the left and right side images
    image_side : list. A list of the side names
    face_side : list. A list of the face side names
    fp : FaceParams. The face parameters
    """

    fp, initialFrame = WhiskerPad.get_nose_tip_coordinates(video_file, video_dir)
    if fp is None or initialFrame is None:
        raise RuntimeError(f"Failed to extract nose tip coordinates from video: {video_file}")
    
    vidcap = cv2.VideoCapture(video_file)
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
                image_side = ['left', 'right']
                
            elif fp.FaceOrientation == 'down':
                image_halves = [vidFrame[:, midWidth:], vidFrame[:, :midWidth]]
                image_side = ['right', 'left']

        elif fp.FaceAxis == 'horizontal':
            midWidth = round(vidFrame.shape[0] / 2)
            if midWidth - vidFrame.shape[0] / 8 < fp.NoseTip[1] < midWidth + vidFrame.shape[0] / 8:
                # If nosetip y value within +/- 1/8 of frame height, use that value
                midWidth = fp.NoseTip[1]
            
            # Get two half images
            if fp.FaceOrientation == 'right':
                image_halves = [vidFrame[:midWidth, :], vidFrame[midWidth:, :]]
                image_side = ['top', 'bottom']
            elif fp.FaceOrientation == 'left':
                image_halves = [vidFrame[midWidth:, :], vidFrame[:midWidth, :]]
                image_side = ['bottom', 'top']

        # By convention, always start with the left side of the face
        face_side = ['left', 'right']

    else:
        image_halves = np.array([vidFrame])
        image_side = ['Full']
        face_side = ['Unknown']

    return image_halves, image_side, face_side, fp

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file", help="Path to the video file")
    parser.add_argument("--splitUp", action="store_true", help="Flag to split the video")
    parser.add_argument("--interactive", action="store_true", help="Flag for interactive mode")
    args = parser.parse_args()

    wp=Params(args.video_file, args.splitUp, args.interactive)

    # Get whisking parameters
    if args.interactive:
        whiskerpadParams, splitUp = WhiskerPad.draw_whiskerpad_roi(wp)
    else:
        whiskerpadParams, splitUp = WhiskerPad.get_whiskerpad_params(wp)

    # Save whisking parameters to json file
    whiskerpad = WhiskerPad.save_whiskerpad_params(wp, whiskerpadParams)

    print(whiskerpad)
