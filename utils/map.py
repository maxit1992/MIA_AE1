import cv2 as cv
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import math


class Map:
    """
    Map class representing a map image, an optional policy image, and utilities for converting between pixel positions
    and geographic coordinates.
    """
    color_map = None
    map = None
    policy = None
    pos_to_coord = None
    coord_to_pos = None

    def __init__(self, path_to_map, path_to_policy=None, calib_coord=None, calib_bmp=None):
        """
        Initialize the Map object. Load the map and policy images, and set up calibration if provided.

        Args:
            path_to_map (str): Path to the map image file.
            path_to_policy (str, optional): Path to the policy image file. Defaults to None.
            calib_coord (array, optional): Calibration coordinates. Defaults to None.
            calib_bmp (array, optional): Calibration bitmap positions. Defaults to None.
        """
        self.color_map = cv.imread(path_to_map)
        self.color_map = cv.cvtColor(self.color_map, cv.COLOR_BGR2RGB)
        if path_to_policy is None:
            self.policy = np.ones(shape=self.map.shape()) * 255
        else:
            self.policy = cv.imread(path_to_policy, cv.IMREAD_GRAYSCALE)
        if calib_coord is not None and calib_bmp is not None:
            self.calibrate(calib_coord, calib_bmp)
        pass

    def calibrate(self, calib_coord, calib_bmp):
        """
        Calibrate the mapping between coordinates and bitmap positions using linear regression.

        Args:
            calib_coord (array): Calibration coordinates.
            calib_bmp (array): Calibration bitmap positions.
        """
        self.coord_to_pos = LinearRegression()
        self.coord_to_pos.fit(calib_coord, calib_bmp)
        self.pos_to_coord = LinearRegression()
        self.pos_to_coord.fit(calib_bmp, calib_coord)

    def to_coord(self, bmp_pos):
        """
        Convert bitmap positions to geographic coordinates.

        Args:
            bmp_pos (list[tuple]): Bitmap positions.

        Returns:
            list[tuple]: Corresponding geographic coordinates.
        """
        if self.pos_to_coord is not None:
            return self.pos_to_coord.predict(bmp_pos)
        else:
            return None

    def to_pos(self, coord):
        """
        Convert geographic coordinates to bitmap positions.

        Args:
            coord (list[tuple]): Geographic coordinates.

        Returns:
            ndarray: Corresponding bitmap positions.
        """
        if self.coord_to_pos is not None:
            return np.round(self.coord_to_pos.predict(coord)).astype(int)
        else:
            return None

    def plot(self, coord=None, file=None):
        """
        Plot the map with optional coordinates marked. If filename is provided, save the plot to the file.

        Args:
            coord (array, optional): Coordinates to mark on the map. Defaults to None.
            file (str, optional): Filename to save the plot. Defaults to None.
        """
        plot_map = self.get_plot(coord)
        if file is not None:
            cv.imwrite(file, cv.cvtColor(plot_map, cv.COLOR_RGB2BGR))
        else:
            plt.figure(figsize=(40, 20))
            plt.grid(False)
            plt.imshow(plot_map)
            plt.show()

    def get_plot(self, coord=None):
        """
        Get the map image with optional coordinates marked.

        Args:
            coord (array, optional): Coordinates to mark on the map. Defaults to None.

        Returns:
            array: The map image with coordinates marked.
        """
        plot_map = self.color_map.copy()
        if coord is not None:
            pos = self.to_pos(coord)
            if pos is not None:
                for point in pos:
                    cv.circle(plot_map, (point[1], point[0]), radius=1, color=(255, 0, 0), thickness=-1)
        return plot_map

    def plot_policy(self):
        """
        Plot the policy image.
        """
        plt.figure(figsize=(40, 20))
        plt.grid(False)
        plt.imshow(self.policy)
        plt.show()

    def get_shape(self):
        """
        Get the shape of the policy image.

        Returns:
            tuple: The shape of the policy image (height, width).
        """
        return self.policy.shape[0:2]

    def get_policy_prob_pos(self, pos):
        """
        Get the policy probabilities for given bitmap positions.

        Args:
            pos (list[tuple]): Bitmap positions.

        Returns:
            list: Policy probabilities for the given positions.
        """
        prob = np.zeros(len(pos))
        for i, point in enumerate(pos):
            if ((point[0] < 0) or (point[0] >= self.policy.shape[0]) or (point[1] < 0) or (
                    point[1] >= self.policy.shape[1])):
                prob[i] = self.policy[0, 0] * 0
            else:
                prob[i] = self.policy[point[0], point[1]] / 255
        return prob

    def get_policy_prob(self, coordinates):
        """
        Get the policy probabilities for given geographic coordinates.

        Args:
            coordinates (list[tuple]): Geographic coordinates.

        Returns:
            list: Policy probabilities for the given coordinates.
        """
        points = self.to_pos(coordinates)
        prob = np.zeros(points.shape[0])
        for i, point in enumerate(points):
            if ((point[0] < 0) or (point[0] >= self.policy.shape[0]) or (point[1] < 0) or (
                    point[1] >= self.policy.shape[1])):
                prob[i] = self.policy[0, 0] * 0
            else:
                prob[i] = self.policy[point[0], point[1]] / 255

        return prob

    @staticmethod
    def destination_point(lat, lon, distance, bearing):
        """
        Returns the destination point from a given point, having travelled the given distance with the given bearing.
        
        Args:
            lat (float): Initial latitude in decimal degrees (e.g. 50.123).
            lon (float): Initial longitude in decimal degrees (e.g. -4.321).
            distance (float): Distance travelled (metres).
            bearing (float): Initial bearing (in degrees from north).
                    
        Returns:
            array: destination point as [latitude,longitude] (e.g. [50.123, -4.321])
        """
        radius = 6371e3  # (Mean) radius of earth

        def to_radians(v):
            return v * math.pi / 180

        def to_degrees(v):
            return v * 180 / math.pi

        delta = distance / radius  # angular distance in radians
        theta = to_radians(bearing)

        phi_1 = to_radians(lat)
        lambda_1 = to_radians(lon)

        sin_phi_1 = math.sin(phi_1)
        cos_phi_1 = math.cos(phi_1)
        sin_delta = math.sin(delta)
        cos_delta = math.cos(delta)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        sin_phi_2 = sin_phi_1 * cos_delta + cos_phi_1 * sin_delta * cos_theta
        phi_2 = math.asin(sin_phi_2)
        y = sin_theta * sin_delta * cos_phi_1
        x = cos_delta - sin_phi_1 * sin_phi_2
        lambda_2 = lambda_1 + math.atan2(y, x)

        lat2 = to_degrees(phi_2)

        lon2 = (to_degrees(lambda_2) + 540) % 360 - 180  # normalise to −180..+180°

        return lat2, lon2

    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in meters between two points on the earth (specified in decimal degrees)
        
        Args:
            lon1 (float): Longitude of point 1.
            lat1 (float): Latitude of point 1.
            lon2 (float): Longitude of point 2.
            lat2 (float): Latitude of point 2.
        Returns:
            float: Distance between the two points in meters.
        """
        # convert decimal degrees to radians
        lon1 = math.radians(lon1)
        lat1 = math.radians(lat1)
        lon2 = math.radians(lon2)
        lat2 = math.radians(lat2)

        # haversine formula
        d_lon = lon2 - lon1
        d_lat = lat2 - lat1
        a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371 * 1000  # Radius of earth in meters.
        return c * r

    @staticmethod
    def get_bearing(lat1, long1, lat2, long2):
        """
        Calculate the bearing between two points.
        
        Args:
            lat1 (float): Latitude of point 1.
            long1 (float): Longitude of point 1.
            lat2 (float): Latitude of point 2.
            long2 (float): Longitude of point 2.
        Returns:
            float: Bearing in degrees.
        """
        d_lon = (long2 - long1)
        x = math.cos(math.radians(lat2)) * math.sin(math.radians(d_lon))
        y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * \
            math.cos(math.radians(lat2)) * math.cos(math.radians(d_lon))
        bearing = np.arctan2(x, y)
        bearing = np.degrees(bearing)

        return bearing
