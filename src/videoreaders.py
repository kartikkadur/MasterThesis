import os
import sys
import cv2
import enum
import numpy as np
#import pyzed.sl as sl
"""
class Images(enum.Enum):
    LEFT = 0
    RIGHT = 1
    LEFT_AND_RIGHT = 2

class SVOReader(object):
    '''Reads a svo encoded video files and returns frames as images/converted video'''
    def __init__(self, fpath, outdir, output='frames', images=Images.LEFT):
        self.filepath = fpath
        self.outdir = outdir
        self.output = output
        self.images = images
        # create directories
        os.makedirs(self.outdir, exist_ok=True)
        self.init_camera()

    def __enter__(self):
        return self

    def init_camera(self):
        # init camera
        self.cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(self.filepath)
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        err = self.cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            sys.stdout.write(repr(err))
            self.cam.close()
            sys.exit(1)
        # set image containers
        self.set_containers()

    def set_containers(self):
        image_size = self.cam.get_camera_information().camera_resolution
        self.width = image_size.width
        self.height = image_size.height

        self.left_image = sl.Mat()
        self.right_image = sl.Mat()
        if self.images == Images.LEFT_AND_RIGHT:
            self.combined_image = np.zeros((self.height, self.width*2, 4), dtype=np.uint8)
        if 'video' in self.output:
            self.video_writer = cv2.VideoWriter(os.path.join(self.outdir, 'video.avi'),
                                       cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                       max(self.cam.get_camera_information().camera_fps, 25),
                                       (self.width, self.height))

    def retrieve_images(self):
        '''retrieves images'''
        if self.images == Images.LEFT:
            self.cam.retrieve_image(self.left_image, sl.VIEW.LEFT)
        elif self.images == Images.RIGHT:
            self.cam.retrieve_image(self.right_image, sl.VIEW.RIGHT)
        elif self.images == Images.LEFT_AND_RIGHT:
            self.cam.retrieve_image(self.left_image, sl.VIEW.LEFT)
            self.cam.retrieve_image(self.right_image, sl.VIEW.RIGHT)

    def write_image(self):
        self.retrieve_images()
        if self.images == Images.LEFT:
            cv2.imwrite(os.path.join(self.outdir, 'left_img_'+str(i)+'.png'), self.left_image.get_data())
        elif self.images == Images.RIGHT:
            cv2.imwrite(os.path.join(self.outdir, 'right_img_'+str(i)+'.png'), self.right_image.get_data())
        elif self.images == Images.LEFT_AND_RIGHT:
            cv2.imwrite(os.path.join(self.outdir, 'left_img_'+str(i)+'.png'), self.left_image.get_data())
            cv2.imwrite(os.path.join(self.outdir, 'right_img_'+str(i)+'.png'), self.right_image.get_data())

    def write_video(self):
        self.retrieve_images()
        if self.images == Images.LEFT:
            img_rgb = cv2.cvtColor(self.left_image.get_data(), cv2.COLOR_RGBA2RGB)
        elif self.images == Images.RIGHT:
            img_rgb = cv2.cvtColor(self.right_image.get_data(), cv2.COLOR_RGBA2RGB)
        elif self.images == Images.LEFT_AND_RIGHT:
            self.combined_image[0:self.height, 0:self.width, :] = self.left_image.get_data()
            self.combined_image[0:, self.width:, :] = self.right_image.get_data()
            img_rgb = cv2.cvtColor(self.combined_image, cv2.COLOR_RGBA2RGB)
        self.video_writer.write(img_rgb)

    def write(self):
        for i in range(self.cam.get_svo_number_of_frames()):
            if self.cam.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
                curr_position = self.cam.get_svo_position()
                if 'image' in self.output:
                    self.write_image()
                else:
                    self.write_video()
    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'video' in self.output:
            self.video_writer.release()
        self.cam.close()
        cv2.destroyAllWindows()
"""
class FrameReader(object):
    """reads the video and returns frames"""
    def __init__(self, fpath):
        self.filepath = fpath
        self.cam = cv2.VideoCapture(self.filepath)
        #if self.cam.isOpened():
        #    self.width  = self.cam.get(cv2.CV_CAP_PROP_FRAME_WIDTH)
        #    self.height = self.cam.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)

    #def get_dimentaion(self):
    #    return (self.width, self.height)

    def __enter__(self):
        return self
    
    def __len__(self):
        return int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    def __getitem__(self, index):
        if index > len(self):
            raise IndexError(f"index {index} is out of range. Max index is {len(self)}")
        if self.cam.isOpened():
            self.cam.set(1, index)
            out, frame = self.cam.read()
            if out:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise RuntimeError('Frame not read. Please check the frame number')
        else:
            raise RuntimeError("Camera is not opened")
        return frame

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'video' in self.output:
            self.videowriter.release()
        self.cam.release()
        cv2.destroyAllWindows()

class FrameWriter(object):
    def __init__(self, fdir, fname='video.avi', output='frames'):
        self.output = output
        self.outdir = fdir
        if 'video' in self.output:
            self.writer = cv2.VideoWriter(os.path.join(fdir, fname),
                                       cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                       25,
                                       (256, 256))
        else:
            self.writer = cv2.imwrite

    def __enter__(self):
        return self

    def write(self, frame, frame_number):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if 'video' in self.output:
            self.writer.write(frame)
        else:
            self.writer.write(os.path.join(self.outdir, f'frame_{frame_number}.png'), frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'video' in self.output:
            self.writer.release()
        cv2.destroyAllWindows()