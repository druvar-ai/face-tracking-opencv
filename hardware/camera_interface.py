class CameraInterface:
    def get_rgb_frame(self):
        raise NotImplementedError

    def get_depth_frame(self):
        raise NotImplementedError
