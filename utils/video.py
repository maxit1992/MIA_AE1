import cv2 as cv


class Video:
    """
    Video class for creating and saving video files from image frames.
    """

    def __init__(self, filename, shape, fps=60):
        """
        Initialize the Video object with the specified filename, shape, and frames per second (fps).

        Args:
            filename (str): The name of the output video file.
            shape (tuple): The shape of the video frames (height, width).
            fps (int, optional): Frames per second for the video. Defaults to 60.
        """
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        self.video = cv.VideoWriter(filename, fourcc, fps, (shape[1], shape[0]))

    def snapshot(self, image):
        """
        Add a frame to the video.

        Args:
            image (array): The image frame to add to the video.
        """
        self.video.write(cv.cvtColor(image, cv.COLOR_RGB2BGR))

    def stop(self):
        """
        Release the video writer and finalize the video file.
        """
        self.video.release()
