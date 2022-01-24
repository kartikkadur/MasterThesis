import sys
import os
import enum
import numpy as np
import cv2
try:
    import pyzed.sl as sl
except:
    print('pyzed was not found. Install it before using this module')
    pass

class Images(enum.Enum):
    LEFT = 0
    RIGHT = 1
    LEFT_AND_RIGHT = 2

class SVOReader(object):
    '''Reads a svo encoded video files and returns frames as images/converted video'''
    def __init__(self, fpath, outdir, output='frames', images=Images.LEFT):
        self.outdir = outdir
        self.output = output
        self.images = images
        if 'video' in output:
            self.fname = os.path.basename(fpath).split(".")[0] if 'video' in output else None
        else:
            self.outdir = os.path.join(outdir, os.path.basename(fpath).split(".")[0])
        # create directories
        os.makedirs(self.outdir, exist_ok=True)
        # init camera
        self.cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(fpath)
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        err = self.cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            sys.stdout.write(repr(err))
            self.cam.close()
            sys.exit(1)
        self.rt_param = sl.RuntimeParameters()
        self.rt_param.sensing_mode = sl.SENSING_MODE.FILL
        image_size = self.cam.get_camera_information().camera_resolution
        self.width = image_size.width
        self.height = image_size.height
        if 'video' in output:
            self.video_writer = cv2.VideoWriter(os.path.join(self.outdir, f'{self.fname}.avi'),
                                       cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                       max(self.cam.get_camera_information().camera_fps, 25),
                                       (self.width, self.height))

    def __enter__(self):
        return self

    def __len__(self):
        return self.cam.get_svo_number_of_frames()

    def get_frame(self):
        image = sl.Mat()
        if self.cam.grab(self.rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = self.cam.get_svo_position()
            # retrieve SVO images
            if self.images == Images.LEFT:
                self.cam.retrieve_image(image, sl.VIEW.LEFT)
            elif self.images == Images.RIGHT:
                self.cam.retrieve_image(image, sl.VIEW.RIGHT)
            elif self.images == Images.LEFT_AND_RIGHT:
                left = sl.Mat()
                right = sl.Mat()
                image = np.zeros((self.height, self.width*2, 4), dtype=np.uint8)
                self.cam.retrieve_image(left, sl.VIEW.LEFT)
                self.cam.retrieve_image(right, sl.VIEW.RIGHT)
                image[0:self.height, 0:self.width, :] = left.get_data()
                image[0:, self.width:, :] = right.get_data()
            if isinstance(image, sl.Mat):
                image = cv2.cvtColor(image.get_data(), cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def write_image(self, image, frame_no):
        if self.images == Images.LEFT:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.outdir, 'left_img_'+str(frame_no)+'.png'), image)
        elif self.images == Images.RIGHT:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.outdir, 'right_img_'+str(frame_no)+'.png'), image)
        elif self.images == Images.LEFT_AND_RIGHT:
            left_image = image[0:self.height, 0:self.width, :]
            right_image = image[0:,self.width:, :]
            left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
            right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.outdir, 'left_img_'+str(frame_no)+'.png'), left_image)
            cv2.imwrite(os.path.join(self.outdir, 'right_img_'+str(frame_no)+'.png'), right_image)

    def write_video(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        self.video_writer.write(img_rgb)

    def write(self, image, frame_no):
        if 'video' in self.output:
            self.write_video(image)
        else:
            self.write_image(image, frame_no)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'video' in self.output:
            self.video_writer.release()
        self.cam.close()
        cv2.destroyAllWindows()

class FrameReader(object):
    """reads the video and returns frames"""
    def __init__(self, fpath):
        self.filepath = fpath
        self.cam = cv2.VideoCapture(self.filepath)

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
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if 'video' in self.output:
            self.writer.write(frame)
        else:
            self.writer.write(os.path.join(self.outdir, f'frame_{frame_number}.png'), frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'video' in self.output:
            self.writer.release()
        cv2.destroyAllWindows()