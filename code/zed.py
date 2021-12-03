import os
import sys
import cv2
import enum
import numpy as np
import pyzed.sl as sl

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

    def convert(self):
        for i in range(self.cam.get_svo_number_of_frames()):
            if self.cam.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
                curr_position = self.cam.get_svo_position()
                if 'image' in self.output:
                    self.write_image()
                else:
                    self.write_video()
        if 'video' in self.output:
            self.video_writer.release()
        self.cam.close()

if __name__ == '__main__':
    svo = SVOReader('/home/kartik/work/DATA/VineYardDATA/25.10.2021/HD720_SN23880_18-11-33.svo', '/home/kartik/work/DATA/VineYardDATA/images', output='video')
    svo.convert()