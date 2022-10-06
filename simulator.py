import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
import scipy.stats as stats
from matplotlib import pyplot
from matplotlib import lines as mlines
import os
import time

def rotationMatrixFromZYXAngles(angles=(0.0, 0.0, 0.0)):

    def Rx(theta):
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])

    def Ry(theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])

    def Rz(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    return Rx(angles[0]) @ Ry(angles[1]) @ Rz(angles[2])

class Camera(object):

    def __init__(self,
                 focal_length=43e-3, # 43mm (35mm optical format)
                 resolution=(4008, 2672), # Kodak KAI-11002 , CCD, color or mono, progressive scan
                 sensor_size=None,
                 pixel_size=(9e-6, 9e-6), # Kodak KAI-11002 , CCD, color or mono, progressive scan
                 fps=5, # Lumenera Lg11059
                 calibration_noise_fn=None,
                 lens_distortion=None,
                 perfect_resolution=False): # True means no quantization

        ''' resolutuion: WxH in pixels
            sensor_size: XxY in meters
            calibration_noise_fn: callable that takes single input which is shape '''

        self._f = focal_length

        if isinstance(resolution, (list, tuple, np.ndarray)):

            self._resolution = np.array(resolution)
        else:
            self._resolution = np.array([resolution, resolution])

        if sensor_size is None:
            if pixel_size is None:
                raise ValueError('Either one of sensor_size or pixel_size should be provided.')

            else:
                if isinstance(pixel_size, (list, tuple, np.ndarray)):

                    self._pixel_size = np.array(pixel_size, dtype=np.float)
                else:
                    self._pixel_size = np.array([pixel_size, pixel_size], dtype=np.float)

        else:
            if isinstance(sensor_size, (list, tuple, np.ndarray)):

                sensor_size = np.array(sensor_size, dtype=np.float)
            else:
                sensor_size = np.array([sensor_size, sensor_size], dtype=np.float)

            self._pixel_size = sensor_size / self._resolution

        self._sensor_size = self._pixel_size * self._resolution

        if calibration_noise_fn is None:
            calibration_noise_fn = lambda pts=None, shape=None: 0.0
        self._calibration_noise_fn = calibration_noise_fn


        self._fps = fps

        self._rotation_matrix = np.eye(3)
        self._translation = np.array([[0., 0., 0.]])

        self._perfect_resolution = perfect_resolution

    @property
    def pixel_size(self):
        return self._sensor_size / self._resolution

    @property
    def resolution(self):
        return self._resolution
    @resolution.setter
    def resolution(self, res):
        self._resolution = res

    @property
    def perfect_resolution(self):
        return self._perfect_resolution

    @perfect_resolution.setter
    def perfect_resolution(self, res):
        self._perfect_resolution = res

    @property
    def euler_rotation_zyx(self):
        return Rotation.from_matrix(self._rotation_matrix).as_euler('zyx')

    @property
    def intrinstic_calibration_matrix(self):

        offset_image_coords = self._resolution / 2.0  # image plane origin in pixels

        f_scales = self._f / self.pixel_size

        # column vector convention
        return np.array([[f_scales[0], 0, offset_image_coords[0]],
                         [0, -f_scales[1], offset_image_coords[1]],
                         [0, 0, 1.0]])

    @property
    def extrinsic_calibration_matrix(self):

        # column vector convention
        return np.concatenate([self._rotation_matrix, self._translation.reshape(3, 1)], axis=1)

    @property
    def calibration_noise(self):
        return self._calibration_noise_fn
    @calibration_noise.setter
    def calibration_noise(self, fn):
        if fn is None:
            self._calibration_noise_fn = lambda pts=None, shape=None: 0.0
        else:
            self._calibration_noise_fn = fn


    @property
    def field_of_view(self):

        # computes field of view (at z = 0)

        img_pts = np.array([[0., 0.],
                            [0., self._resolution[1] - 1],
                            [self._resolution[0] - 1, self._resolution[1] - 1],
                            [self._resolution[0] - 1, 0.]])

        z = np.zeros(shape=(4, 1))

        return self.backProject3D(img_pts, z)

    def backProject3D(self, img_points, z_coords):

        '''for given z_coords, this function back-projects the continuous image points to world X-Y coordinates.
        useful for sampling within field of view or to compute field of view for specific depth.'''

        img_points = img_points.reshape(-1, 2, 1)
        z_coords = z_coords.reshape(-1, 1, 1)
        z = np.concatenate([z_coords, np.ones_like(z_coords)], axis=1)

        P = self.intrinstic_calibration_matrix @ self.extrinsic_calibration_matrix

        # based on derivations on the paper, below part exploits broadcasting for parallel computing

        u = P[2][:2].reshape(1, 1, 2)
        v = P[2][2:].reshape(1, 1, 2)
        K_l = P[:2, :2].reshape(1, 2, 2)
        K_r = P[:2, 2:].reshape(1, 2, 2)

        A = img_points @ u - K_l
        b = (K_r - img_points @ v) @ z

        w = np.linalg.inv(A) @ b

        return w.reshape(-1, 2)


    def captureInterestPoints(self, interest_points):
        '''gets points in world coordinates and returns two-tuple consisting of projections onto image plane
        and a mask for the points in the field of view'''

        num_pts = interest_points.shape[0]

        # first transform from world coordinates to pixel coordinates
        T = self._translation
        R = self._rotation_matrix

        interest_points_in_p = ((interest_points @ R.T) + T) @ self.intrinstic_calibration_matrix.T

        interest_points_in_p = interest_points_in_p[:, :2] / interest_points_in_p[:, 2].reshape(-1, 1)

        # if location dependent noise is desired then pass pts instead of shape
        interest_points_in_p = interest_points_in_p + self.calibration_noise(shape=(num_pts,))

        points_in_fov = np.equal(
            np.sum(np.logical_and(
                np.greater(interest_points_in_p, 0.),
                np.less(interest_points_in_p, self._resolution.reshape(1, 2))).astype(np.float),
                   axis=1), 2)

        # mask out points out of  fov
        interest_points_in_p = interest_points_in_p * points_in_fov.astype(np.float).reshape(-1, 1)

        # quantization
        if not self._perfect_resolution:
            interest_points_in_p = interest_points_in_p.astype(np.int).astype(np.float)

        return interest_points_in_p, points_in_fov

class UAV(object):

    def __init__(self,
                 rotation=(np.pi, 0.0, 0.0), # (roll(x), pitch(y), yaw(z)) in the order zyx
                 coords=(0.0, 0.0, 5e3), # 5000m elevation
                 speed=200, # in km/h
                 camera=None,
                 camera_rotation=(0.0, 0.0, np.pi/2), # (roll(x), pitch(y), yaw(z)) in the order zyx wrt plane origin
                 camera_coords=(3.5, 0.0, 0.5), # xyz wrt plane origin
                 has_gimbal=False,
                 gps_noise_fn=None):

        if isinstance(rotation, (list, tuple, np.ndarray)):
            self._rotation = np.array(rotation)
        else:
            raise ValueError('rotation should be 3-D vector.')

        if isinstance(coords, (list, tuple, np.ndarray)):
            self._coords = np.array(coords)
        else:
            raise ValueError('coords should be 3-D vector.')

        self._speed = speed * 1000. / 3600. # km/h to m/s

        if isinstance(camera, Camera):
            self._camera = camera
        else:
            raise ValueError('camera should be an instance of Camera class.')

        if isinstance(camera_rotation, (list, tuple, np.ndarray)):
            self._camera_rotation = np.array(camera_rotation)
        else:
            raise ValueError('camera_rotation should be 3-D vector.')

        if isinstance(camera_coords, (list, tuple, np.ndarray)):
            self._camera_coords = np.array(camera_coords)
        else:
            raise ValueError('camera_coords should be 3-D vector.')

        if gps_noise_fn is None:
            gps_noise_fn = lambda pts=None, shape=None: 0.0
        self._gps_noise_fn = gps_noise_fn

        self._has_gimbal = has_gimbal


        # update camera
        self.updateCameraPose()

    @property
    def gps_noise(self):
        return self._gps_noise_fn

    @gps_noise.setter
    def gps_noise(self, fn):
        self._gps_noise_fn = fn

    @property
    def camera(self):
        return self._camera

    @property
    def rotation_matrix(self):
        return rotationMatrixFromZYXAngles(angles=self._rotation)


    @property
    def coords(self):
        return self._coords

    @property
    def euler_rotation_zyx(self):
        return Rotation.from_matrix(self.rotation_matrix).as_euler('zyx')[::-1]

    @property
    def absolute_camera_coords(self):
        return self._coords + self.rotation_matrix.T @ self._camera_coords

    @property
    def relative_camera_rotation_matrix(self):
        if self._has_gimbal:
            Rz = rotationMatrixFromZYXAngles(angles=(0.0, 0.0, self._rotation[2]))
            Rc = rotationMatrixFromZYXAngles(angles=self._camera_rotation)
            return Rc @ Rz @ self.rotation_matrix.T
        else:
            return rotationMatrixFromZYXAngles(angles=self._camera_rotation)


    def updateCameraPose(self):

        # absolute camera coord is (plane coords in world) + inv(R_plane) @ (camera coords in plane)
        # then translation is -R_cam@R_plane(absolute camera coord) = -R_cam@R_plane@(plane coords in world) + R_cam@(camera coords in plane)

        # camera rotation in column vector convention
        self._camera._rotation_matrix = self.relative_camera_rotation_matrix @ self.rotation_matrix

        # T = -RC
        self._camera._translation = \
            -(self._coords.reshape(1, 3) @ self.rotation_matrix.T +
              self._camera_coords.reshape(1, 3)) @ self.relative_camera_rotation_matrix.T

    def activateGimbal(self):
        self._has_gimbal = True

    def deactivateGimbal(self):
        self._has_gimbal = False

    def applyControl(self, droll_dt, dpitch_dt, dyaw_dt, duration, dt=1e-6, radians=True):
        ''' plane is moving in the x direction of the plane coordinates '''

        d_angles = np.array((droll_dt, dpitch_dt, dyaw_dt))

        if not radians:
            d_angles = d_angles / 180. * np.pi

        num_steps = np.round(duration / dt).astype(np.int)

        # we need inverse rotation matrix
        # note that R = Rx(roll)Ry(pitch)Rz(yaw) thus R_inv = Rz(-yaw)Ry(-pitch)Rx(-roll)
        R_rel = rotationMatrixFromZYXAngles(angles=d_angles)
        C = R_rel.T @ np.array((self._speed * dt, 0., 0.)).reshape(3, 1) #
        T = np.zeros(shape=(3, 1))
        R = np.eye(3)

        #T_list = [T]
        #R_list = [np.zeros(3)]
        for k in range(num_steps):

            # T_next = C + R_inv @ T_prev
            T = R_rel.T @ T + C

            R = R_rel @ R

            #T_list.append(T)
            #R_list.append(Rotation.from_matrix(R).as_euler('zyx')[::-1])


        relative_rotation = Rotation.from_matrix(R).as_euler('zyx')[::-1]
        relative_translation = - (R @ T).reshape(1, 3)

        cam2cam_translation = \
            self.relative_camera_rotation_matrix @ R @ (R.T @ self._camera_coords.reshape(3, 1) + T
                                                        - self._camera_coords.reshape(3, 1))
        cam2cam_translation = - cam2cam_translation.reshape(1, 3)

        rotation = Rotation.from_matrix(R @ self.rotation_matrix).as_euler('zyx')[::-1].reshape(1, 3)


        coords = self._coords + (self.rotation_matrix.T @ T).reshape(-1)

        self.setPose(rotation=rotation, coords=coords)

        return relative_rotation, relative_translation, cam2cam_translation#, R_list, T_list


    def setPose(self,
                rotation=(0.0, 0.0, 0.0),  # (roll(x), pitch(y), yaw(z)) in the order zyx
                coords=(0.0, 0.0, 5e3)):  # 5000m elevation

        ''' Rotation order is z->y->x ie. R = R(x)R(y)R(z) '''

        self._rotation = np.array(rotation).reshape(-1)
        self._coords = np.array(coords)
        self.updateCameraPose()


    def estimatePose(self, interest_points, method='epnp'):

        method_dict = {'epnp': cv.SOLVEPNP_EPNP,
                       'p3p': cv.SOLVEPNP_P3P,
                       'dlt': cv.SOLVEPNP_ITERATIVE,
                       'dls': cv.SOLVEPNP_DLS}

        method = method.lower()

        if method in method_dict:
            pnp_method = method_dict[method]
        else:
            print('warning: method {} is not in {}, thus default method {} is to be used!'.format(method,
                                                                                                  method_dict.keys(),
                                                                                                  'p3p'))
            pnp_method = method_dict['p3p']

        noisy_interest_points = interest_points + self.gps_noise(shape=interest_points.shape[0])

        image_points, val_mask = self._camera.captureInterestPoints(interest_points=noisy_interest_points)

        if np.sum(val_mask) < 6:
            success = False
            result = None
        else:
            solverPnP = cv.solvePnPRansac if method == 'p3p' else cv.solvePnP
            result = solverPnP(objectPoints=interest_points[val_mask].astype(np.float32),
                               imagePoints=image_points[val_mask].astype(np.float32),
                               cameraMatrix=self._camera.intrinstic_calibration_matrix,
                               distCoeffs=np.zeros((4, 1)),  # assuming no lens distortion
                               flags=pnp_method)

            success = result[0]

        if not success:
            estimated_euler_rotation = None
            estimated_coords = None
        else:
            rot_vector = result[1].reshape(-1)
            translation_vector = result[2].reshape(1, 3)

            R_cam = Rotation.from_rotvec(rot_vector).as_matrix()
            coords_cam = - translation_vector @ R_cam
            estimated_coords = coords_cam.reshape(-1) - self._camera_coords

            R_uav = self.relative_camera_rotation_matrix.T @ R_cam
            estimated_euler_rotation = Rotation.from_matrix(R_uav).as_euler('zyx')[::-1]


        return success, estimated_euler_rotation, estimated_coords

    def estimateRelativePose(self, points_in_1, points_in_2, method='ransac'):

        method_dict = {'ransac': cv.RANSAC,
                       'lmeds': cv.LMEDS}

        method = method.lower()

        if method in method_dict:
            emat_method = method_dict[method]
        else:
            print('warning: method {} is not in {}, thus default method {} is to be used!'.format(method,
                                                                                                  method_dict.keys(),
                                                                                                  'ransac'))
            emat_method = method_dict['ransac']

        num_pts = points_in_1.shape[0]

        if num_pts < 5:
            success = False
            ematrix = None
            inlier_mask = None
        else:
            ematrix, inlier_mask = cv.findEssentialMat(points1=points_in_1.astype(np.float32),
                                                       points2=points_in_2.astype(np.float32),
                                                       cameraMatrix=self._camera.intrinstic_calibration_matrix,
                                                       method=emat_method,
                                                       prob=0.9999,
                                                       threshold=1e-3)

            if ematrix is not None:
                success = ((ematrix.shape[0] == 3) and (ematrix.shape[1] == 3))
            else:
                success = False

        if not success:
            estimated_rotation = None
            estimated_translation_direction = None
        else:

            _, R_cam, T_cam, _ = cv.recoverPose(E=ematrix,
                                                points1=points_in_1.astype(np.float32),
                                                points2=points_in_2.astype(np.float32),
                                                cameraMatrix=self._camera.intrinstic_calibration_matrix,
                                                mask=inlier_mask)

            R_rel = \
                self.relative_camera_rotation_matrix.T @ R_cam @ self.relative_camera_rotation_matrix

            estimated_translation_direction = T_cam.reshape(1, 3)
            estimated_rotation = Rotation.from_matrix(R_rel).as_euler('zyx')[::-1].reshape(1, 3)

        return success, estimated_rotation, estimated_translation_direction


class Simulator(object):

    def __init__(self,
                 uav_args=None,
                 cam_args=None,
                 gps_errors={'low': (0.35, 0.5) , 'medium': (2, 3.5), 'high': (5, 8)}, # in meters
                 calibration_errors={'low': 0.5 , 'medium': 1, 'high': 4}, # in pixel size
                 elevation_ranges={'small': 10, 'large': 100}, # in meters

                 base_elevation=460,
                 image_plane_boundary_tolerance=100,
                 samples_per_frame=15,
                 iterations_per_setting=100):

        self._uav = UAV(**uav_args, camera=Camera(**cam_args))

        self._gps_errors = gps_errors
        self._calibration_errors = calibration_errors
        self._elevation_ranges = elevation_ranges
        self._base_elevation = base_elevation
        self._image_plane_boundary_tolerance = image_plane_boundary_tolerance
        self._samples_per_frame = samples_per_frame
        self._iterations_per_setting = iterations_per_setting


    def calibrationNoise(self, level='zero'):

        if level == 'zero':
            return lambda pts=None, shape=None: 0.0
        else:
            mu = np.array([0., 0])
            std = self._calibration_errors[level]
            cov = np.diag([std**2, std**2])

            return lambda shape: np.random.multivariate_normal(mean=mu,
                                                               cov=cov,
                                                               size=shape)

    def gpsNoise(self, level='zero'):

        if level == 'zero':
            return lambda pts=None, shape=None: 0.0
        else:
            mu_xy = np.array([0., 0.])
            std_xy = self._gps_errors[level][0]
            cov_xy = np.diag([std_xy**2, std_xy**2])

            mu_z = 0.
            std_z = self._gps_errors[level][1]

            return lambda shape: np.concatenate([np.random.multivariate_normal(mean=mu_xy, cov=cov_xy, size=shape),
                                                 np.expand_dims(np.random.normal(loc=mu_z, scale=std_z, size=shape), axis=-1)],
                                                axis=-1)

    def sample3DPoints(self, num_samples, z_range_level='large'):
        tol = self._image_plane_boundary_tolerance
        x_max = self._uav.camera.resolution[0] - 1 - tol
        y_max = self._uav.camera.resolution[1] - 1 - tol

        x_img = np.random.uniform(low=tol, high=x_max, size=(num_samples, 1))
        y_img = np.random.uniform(low=tol, high=y_max, size=(num_samples, 1))
        z_world = self._base_elevation + \
                  np.random.uniform(low=0,
                                    high=self._elevation_ranges[z_range_level],
                                    size=(num_samples, 1))

        xy_world = \
            self._uav.camera.backProject3D(img_points=np.concatenate([x_img, y_img], axis=-1),
                                           z_coords=z_world)

        world_coords = np.concatenate([xy_world, z_world], axis=-1)

        return world_coords

    # PnP related
    # =============================================================================
    def estimatePnPComputationTime(self):

        self._uav.gps_noise = self.gpsNoise(level='zero')
        self._uav.camera.calibration_noise = self.calibrationNoise(level='zero')
        pts3d = self.sample3DPoints(num_samples=self._samples_per_frame, z_range_level='large')

        time_elapsed = {'epnp': 0, 'p3p': 0, 'dlt': 0, 'dls': 0}

        method_dict = {'epnp': cv.SOLVEPNP_EPNP,
                       'p3p': cv.SOLVEPNP_P3P,
                       'dlt': cv.SOLVEPNP_ITERATIVE,
                       'dls': cv.SOLVEPNP_DLS}


        image_points, val_mask = \
            self._uav.camera.captureInterestPoints(interest_points=pts3d)

        calib_mat = self._uav.camera.intrinstic_calibration_matrix

        object_points = pts3d[val_mask].astype(np.float32)
        image_points = image_points[val_mask].astype(np.float32)
        dist_coeffs = np.zeros((4, 1))

        for method in time_elapsed:
            solverPnP = cv.solvePnPRansac if method == 'p3p' else cv.solvePnP
            pnp_method = method_dict[method]
            beg = time.time()
            for k in range(self._iterations_per_setting):
                _ = solverPnP(objectPoints=object_points,
                              imagePoints=image_points,
                              cameraMatrix=calib_mat,
                              distCoeffs=dist_coeffs,
                              flags=pnp_method)

            te = (time.time() - beg) / self._iterations_per_setting

            time_elapsed[method] = te * 1e3

        return time_elapsed

    def plotPnPComputationTime(self):

        epnp = []
        p3p = []
        dlt = []
        dls = []
        num_samples = [9, 11, 15, 21, 27, 35, 45, 63, 81, 117]
        for n in num_samples:
            self._samples_per_frame = n
            time_elapsed = self.estimatePnPComputationTime()
            epnp.append(time_elapsed['epnp'])
            p3p.append(time_elapsed['p3p'])
            dlt.append(time_elapsed['dlt'])
            dls.append(time_elapsed['dls'])

        pyplot.plot(num_samples, epnp, color='r', marker='o', label='epnp', linewidth=2)
        pyplot.plot(num_samples, p3p, color='b', marker='s', label='p3p', linewidth=2)
        pyplot.plot(num_samples, dlt, color='m', marker='^', label='dlt', linewidth=2)
        pyplot.plot(num_samples, dls, color='brown', marker='v', label='dls', linewidth=2)

        pyplot.ylabel('time elapsed (ms)')
        pyplot.xlabel('number of points')

        pyplot.title('computation time comparison')

        pyplot.legend(loc='best')

        pyplot.show()

    def estimatePnPErrorDistribution(self,
                                     resolution_scaler=1.,
                                     gps_error_level='zero',
                                     calibration_error_level='zero',
                                     elevation_range='large',
                                     method='epnp'):

        original_resolution = self._uav.camera.resolution
        self._uav.camera.resolution = resolution_scaler * original_resolution
        self._uav.gps_noise = self.gpsNoise(level=gps_error_level)
        self._uav.camera.calibration_noise = self.calibrationNoise(level=calibration_error_level)

        errs_angles = []
        errs_coords = []
        failures = []
        for k in range(self._iterations_per_setting):

            num_fails = -1
            success = False
            angles = np.zeros(3)
            coords = np.zeros(3)
            while not success:
                num_fails += 1
                if num_fails > 100:
                    print('warning: PnP has failed more than 100 times!')
                    break

                pts3d = self.sample3DPoints(num_samples=self._samples_per_frame,
                                            z_range_level=elevation_range)

                success, angles, coords = self._uav.estimatePose(pts3d, method=method)

            if success:
                errs_angles.append(np.expand_dims(angles - self._uav.euler_rotation_zyx, axis=0))
                errs_coords.append(np.expand_dims(coords - self._uav.coords, axis=0))
            failures.append(num_fails)

        errs_angles = np.concatenate(errs_angles, axis=0)
        errs_coords = np.concatenate(errs_coords)
        failures = np.array(failures).astype(np.float)

        res_text = 'hd' if resolution_scaler < 1 else 'uhd'
        rot = self._uav.euler_rotation_zyx * 180 / np.pi
        coords = self._uav.coords
        setting_text = \
            r'aircraf pose: ({:.2f},{:.2f},{:.2f})$\circ$, ({:.2f},{:.2f},{:.2f})m'.format(
                rot[0], rot[1], rot[2], coords[0], coords[1], coords[2]) + '\n' + \
            'res.: {} | 3d noise: {} | calib. noise: {} | z range: {} | method: {}'.format(
                res_text, gps_error_level, calibration_error_level, elevation_range, method)

        fname = '{}_{}_{}_{}_{}.png'.format(method, gps_error_level, calibration_error_level, elevation_range, res_text)

        # restore resolution
        self._uav.camera.resolution = original_resolution

        return errs_angles, errs_coords, failures, setting_text, fname

    def plotPnPErrorDistribution(self, errs_angles, errs_coords, num_failures, setting_text, fname):

        def plot_pdf(mu, sigma, ax, linewidth=3):
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma),
                    label=r'$\mu$={:.2f}, $\sigma$={:.2f}'.format(mu, sigma),
                    color='red',
                    linewidth=linewidth)

        avg_num_failures = np.round(np.mean(num_failures)).astype(np.int)

        errs_angles = errs_angles * 180. / np.pi # to degrees

        errs_pose = np.concatenate([errs_angles, errs_coords], axis=-1)

        mean_errs = np.mean(errs_pose, axis=0)
        std_errs = np.std(errs_pose, axis=0)

        labels = (r'$\alpha$', r'$\beta$', r'$\gamma$', 'x', 'y', 'z')
        titles = (r'$\alpha - \alpha_{true}$',
                  r'$\beta - \beta_{true}$',
                  r'$\gamma - \gamma_{true}$',
                  r'$x - x_{true}$',
                  r'$y - y_{true}$',
                  r'$z - z_{true}$')

        x_labels = (r'x rotation error ($\circ$)',
                    r'y rotation error ($\circ$)',
                    r'z rotation error ($\circ$)',
                    r'x position error (m)',
                    r'y position error (m)',
                    r'z position error (m)')

        linewidth = 2
        bins = 100
        dpi = 96 #
        fig_size = (1440/dpi, 960/dpi)
        fig, axs = pyplot.subplots(2, 3, figsize=fig_size, dpi=dpi)
        axs = axs.reshape(-1)

        for k in range(6):
            ax = axs[k]

            # plot histogram
            _ = ax.hist(errs_pose[:, k], bins=bins, density=True, histtype='step',
                        range=(mean_errs[k] - 6 * std_errs[k], mean_errs[k] + 6 * std_errs[k]),
                        label=r'{}-err. hist.'.format(labels[k]),
                        linewidth=linewidth, color='blue')
            plot_pdf(mean_errs[k], std_errs[k], ax, linewidth=linewidth)

            ax.set_title(titles[k])
            ax.set_xlabel(x_labels[k])
            ax.set_ylabel('density')
            ax.set_xlim(left=mean_errs[k]-10*std_errs[k])

            # prepare legend
            h, lbls = ax.get_legend_handles_labels()
            h, lbls = zip(*sorted(zip(h, lbls), key=lambda t: t[0].__class__.__name__ == 'Line2D'))
            handles = \
                [mlines.Line2D([], [], color='b', marker='None',
                               linestyle='-', linewidth=linewidth,
                               label=lbls[0]),
                 mlines.Line2D([], [], color='r', marker='None',
                               linestyle='-', linewidth=linewidth,
                               label=lbls[1])]#,
                 #mlines.Line2D([], [], color='k', marker='None', linestyle='None',
                  #             label='num. fails.: {}'.format(avg_num_failures))]
            ax.legend(handles=handles, loc='upper left')



        fig.suptitle(setting_text)
        fig.savefig(fname)
        pyplot.close(fig)

    def runPnPSimulations(self, save_path):

        def fname_prefix(rot, has_gimbal):
            rot_x = 'x' if rot[0] != 0.0 else ''
            rot_y = 'y' if rot[1] != 0.0 else ''
            rot_z = 'z' if rot[2] != 0.0 else ''
            gimbal_text = '_gimbal_' if has_gimbal else '_'
            prefix = rot_x + rot_y + rot_z + gimbal_text
            return prefix


        methods = ('epnp', 'p3p', 'dlt', 'dls')

        rotations = ((0, 0, 0),
                     (0, 0, np.pi/3),
                     (0, -np.pi/15, np.pi/3),
                     (-np.pi/18, -np.pi/15, np.pi/3))

        rotations = [rotations[-1]]

        gps_noise_levels = ('low', 'medium', 'high')

        calibration_noise_levels = ('low', 'medium', 'high')

        elevation_range = ('large', )#('small', 'large')

        resolution_scaler = (1, )#(0.5, 1)

        gimbal_controls = (True, ) #(False, True)

        for method in methods:
            for z_range in elevation_range:
                for gimbal in gimbal_controls:
                    if gimbal:
                        self._uav.activateGimbal()
                    else:
                        self._uav.deactivateGimbal()
                    for rot in rotations:
                        self._uav.setPose(rotation=rot, coords=self._uav.coords)
                        for gps_noise in gps_noise_levels:
                            for calibration_noise in calibration_noise_levels:
                                print('running: {}:{}:{}:{}:{}:g{}'.format(method, z_range, rot, gps_noise, calibration_noise, gimbal))
                                errs_angles, errs_coords, failures, text, fname = self.estimatePnPErrorDistribution(
                                    gps_error_level=gps_noise,
                                    calibration_error_level=calibration_noise,
                                    elevation_range=z_range,
                                    resolution_scaler=1,
                                    method=method)
                                fname = fname_prefix(rot, gimbal) + fname

                                self.plotPnPErrorDistribution(errs_angles, errs_coords, failures, text,
                                                           os.path.join(save_path, fname))

                        # resolution tests
                        for s in resolution_scaler:
                            print('running: {}:{}:{}:{}:{}:{}'.format(method, z_range, rot, 'zero', 'zero', s))
                            errs_angles, errs_coords, failures, text, fname = self.estimatePnPErrorDistribution(
                                gps_error_level='zero',
                                calibration_error_level='zero',
                                elevation_range=z_range,
                                resolution_scaler=s,
                                method=method)
                            fname = fname_prefix(rot, gimbal) + fname

                            self.plotPnPErrorDistribution(errs_angles, errs_coords, failures, text,
                                                       os.path.join(save_path, fname))

    # E-Matrix Related
    # ===============================================================================
    def estimateEMatrixComputationTime(self):

        control_input = {'droll_dt': -10. / 200,
                         'dpitch_dt': -5. / 200,
                         'dyaw_dt': 30. / 200,
                         'duration': 2,
                         'dt': 1e-2,
                         'radians': False}

        self._uav.gps_noise = self.gpsNoise(level='zero')
        self._uav.camera.calibration_noise = self.calibrationNoise(level='zero')
        pts3d = self.sample3DPoints(num_samples=self._samples_per_frame, z_range_level='large')

        image_points_1, val_mask_1 = \
            self._uav.camera.captureInterestPoints(interest_points=pts3d)

        _ = self._uav.applyControl(**control_input)

        image_points_2, val_mask_2 = \
            self._uav.camera.captureInterestPoints(interest_points=pts3d)

        val_mask = np.logical_and(val_mask_1, val_mask_2)
        points_in_1 = image_points_1[val_mask]
        points_in_2 = image_points_2[val_mask]

        time_elapsed = {'lmeds': 0, 'ransac': 0}

        method_dict = {'lmeds': cv.LMEDS,
                       'ransac': cv.RANSAC}

        calib_mat = self._uav.camera.intrinstic_calibration_matrix


        for method in time_elapsed:

            emat_method = method_dict[method]
            beg = time.time()
            for k in range(self._iterations_per_setting):
                _ = cv.findEssentialMat(points1=points_in_1.astype(np.float32),
                                        points2=points_in_2.astype(np.float32),
                                        cameraMatrix=calib_mat,
                                        method=emat_method,
                                        prob=0.9999,
                                        threshold=1e-3)

            te = (time.time() - beg) / self._iterations_per_setting

            time_elapsed[method] = te * 1e3

        return time_elapsed

    def plotEMatrixComputationTime(self):

        lmeds = []
        ransac = []
        num_samples = [9, 11, 15, 21, 27, 35, 45, 63, 81, 117]
        for n in num_samples:
            self._samples_per_frame = n
            time_elapsed = self.estimateEMatrixComputationTime()
            lmeds.append(time_elapsed['lmeds'])
            ransac.append(time_elapsed['ransac'])

        pyplot.plot(num_samples, lmeds, color='r', marker='o', label='lmeds', linewidth=2)
        pyplot.plot(num_samples, ransac, color='b', marker='s', label='ransac', linewidth=2)

        pyplot.ylabel('time elapsed (ms)')
        pyplot.xlabel('number of points')

        pyplot.title('computation time comparison')

        pyplot.legend(loc='best')

        pyplot.show()

    def estimateEMatrixErrorDistribution(self,
                                         control_input,
                                         resolution_scaler=1.,
                                         gps_error_level='zero',
                                         calibration_error_level='zero',
                                         elevation_range='large',
                                         method='lmeds',
                                         reject_outliers=True):

        original_resolution = self._uav.camera.resolution
        original_quantization = self._uav.camera.perfect_resolution
        self._uav.camera.resolution = resolution_scaler * original_resolution
        self._uav.gps_noise = self.gpsNoise(level=gps_error_level)
        self._uav.camera.calibration_noise = self.calibrationNoise(level=calibration_error_level)

        initial_rot = self._uav.euler_rotation_zyx
        initial_coords = self._uav.coords

        errs_angles = []
        errs_coords = []
        failures = []
        for k in range(self._iterations_per_setting):
            num_fails = -1
            success = False
            true_rotation = estimated_rotation = np.zeros(shape=(1, 3))
            true_cam2cam_translation = estimated_translation_direction = np.zeros(shape=(1, 3))
            while not success:
                num_fails += 1
                if num_fails > 100:
                    print('warning: EMatrix has failed more than 100 times!')
                    break

                self._uav.setPose(rotation=initial_rot, coords=initial_coords)

                pts3d = self.sample3DPoints(num_samples=self._samples_per_frame,
                                            z_range_level=elevation_range)

                noisy_interest_points = pts3d + self._uav.gps_noise(shape=pts3d.shape[0])

                if reject_outliers:
                    self._uav.camera.calibration_noise = None
                    self._uav.camera.perfect_resolution = True
                    image_points_1, val_mask_1 = \
                        self._uav.camera.captureInterestPoints(interest_points=noisy_interest_points)

                    true_rotation, _, _ = self._uav.applyControl(**control_input)

                    image_points_2, val_mask_2 = \
                        self._uav.camera.captureInterestPoints(interest_points=noisy_interest_points)

                    val_mask = np.logical_and(val_mask_1, val_mask_2)

                    ok, estimated_rotation, _ = \
                        self._uav.estimateRelativePose(points_in_1=image_points_1[val_mask],
                                                       points_in_2=image_points_2[val_mask],
                                                       method=method)
                    is_outlier = \
                        np.sum(np.abs(np.degrees(estimated_rotation - true_rotation)) > 1e-3).astype(bool) \
                            if ok else True

                    # restore camera
                    self._uav.camera.calibration_noise = self.calibrationNoise(level=calibration_error_level)
                    self._uav.camera.perfect_resolution = original_quantization
                    self._uav.setPose(rotation=initial_rot, coords=initial_coords)
                else:
                    is_outlier = False

                if not is_outlier:

                    image_points_1, val_mask_1 = \
                        self._uav.camera.captureInterestPoints(interest_points=noisy_interest_points)

                    true_rotation, true_translation, true_cam2cam_translation = self._uav.applyControl(**control_input)

                    image_points_2, val_mask_2 = \
                        self._uav.camera.captureInterestPoints(interest_points=noisy_interest_points)

                    val_mask = np.logical_and(val_mask_1, val_mask_2)

                    success, estimated_rotation, estimated_translation_direction = \
                        self._uav.estimateRelativePose(points_in_1=image_points_1[val_mask],
                                                   points_in_2=image_points_2[val_mask],
                                                   method=method)

            if success:
                true_translation_mag = np.sqrt(np.sum(np.square(true_cam2cam_translation)))
                estimated_translation = estimated_translation_direction * true_translation_mag

                errs_angles.append(estimated_rotation - true_rotation)
                errs_coords.append(estimated_translation - true_cam2cam_translation)

            failures.append(num_fails)

        errs_angles = np.concatenate(errs_angles, axis=0)
        errs_coords = np.concatenate(errs_coords, axis=0)
        failures = np.array(failures).astype(np.float)

        res_text = 'hd' if resolution_scaler < 1 else 'uhd'



        t = - (rotationMatrixFromZYXAngles(angles=true_rotation).T @ true_translation.reshape(3, 1)).reshape(-1)
        rot = np.degrees(true_rotation)

        setting_text = \
            r'$\Delta$ aircraft pose: ({:.2f},{:.2f},{:.2f})$\circ$, ({:.2f},{:.2f},{:.2f})m'.format(
                rot[0], rot[1], rot[2], t[0], t[1], t[2]) + '\n' + \
            'res.: {} | 3d noise: {} | calib. noise: {} | z range: {} | method: {} | # fails: {}'.format(
                res_text, gps_error_level, calibration_error_level, elevation_range, method,
                np.sum(failures).astype(np.int))

        fname = '{}_{}_{}_{}_{}.png'.format(method, gps_error_level, calibration_error_level, elevation_range, res_text)

        # restore resolution
        self._uav.camera.resolution = original_resolution

        # restore pose
        self._uav.setPose(rotation=initial_rot, coords=initial_coords)

        return errs_angles, errs_coords, failures, setting_text, fname


    def plotEMatrixErrorDistribution(self, errs_angles, errs_coords, num_failures, setting_text, fname):

        def plot_pdf(mu, sigma, ax, linewidth=3):
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma),
                    label=r'$\mu$={:.2f}, $\sigma$={:.2f}'.format(mu, sigma),
                    color='red',
                    linewidth=linewidth)

        avg_num_failures = np.round(np.mean(num_failures)).astype(np.int)

        errs_angles = np.degrees(errs_angles) # to degrees

        errs_pose = np.concatenate([errs_angles, errs_coords], axis=-1)

        mean_errs = np.mean(errs_pose, axis=0)
        std_errs = np.std(errs_pose, axis=0)

        labels = (r'$\alpha$', r'$\beta$', r'$\gamma$', 'x', 'y', 'z')
        titles = (r'$\alpha - \alpha_{true}$',
                  r'$\beta - \beta_{true}$',
                  r'$\gamma - \gamma_{true}$',
                  r'$x - x_{true}$',
                  r'$y - y_{true}$',
                  r'$z - z_{true}$')

        x_labels = (r'x rotation error ($\circ$)',
                    r'y rotation error ($\circ$)',
                    r'z rotation error ($\circ$)',
                    r'x position error (m)',
                    r'y position error (m)',
                    r'z position error (m)')

        linewidth = 2
        bins = 100
        dpi = 96 #
        fig_size = (1440/dpi, 960/dpi)
        fig, axs = pyplot.subplots(2, 3, figsize=fig_size, dpi=dpi)
        axs = axs.reshape(-1)

        for k in range(6):
            ax = axs[k]

            # plot histogram
            _ = ax.hist(errs_pose[:, k], bins=bins, density=True, histtype='step',
                        range=(mean_errs[k] - 6 * std_errs[k], mean_errs[k] + 6 * std_errs[k]),
                        label=r'{}-err. hist.'.format(labels[k]),
                        linewidth=linewidth, color='blue')
            plot_pdf(mean_errs[k], std_errs[k], ax, linewidth=linewidth)

            ax.set_title(titles[k])
            ax.set_xlabel(x_labels[k])
            ax.set_ylabel('density')
            ax.set_xlim(left=mean_errs[k]-10*std_errs[k])

            # prepare legend
            h, lbls = ax.get_legend_handles_labels()
            h, lbls = zip(*sorted(zip(h, lbls), key=lambda t: t[0].__class__.__name__ == 'Line2D'))
            handles = \
                [mlines.Line2D([], [], color='b', marker='None',
                               linestyle='-', linewidth=linewidth,
                               label=lbls[0]),
                 mlines.Line2D([], [], color='r', marker='None',
                               linestyle='-', linewidth=linewidth,
                               label=lbls[1])]#,
                 #mlines.Line2D([], [], color='k', marker='None', linestyle='None',
                  #             label='num. fails.: {}'.format(avg_num_failures))]
            ax.legend(handles=handles, loc='upper left')

        fig.suptitle(setting_text)
        fig.savefig(fname)
        pyplot.close(fig)

    def runEMatrixSimulations(self, save_path, reject_outliers=True):

        def fname_prefix(rot, dur, pts):
            rot_x = 'x' if rot[0] != 0.0 else ''
            rot_y = 'y' if rot[1] != 0.0 else ''
            rot_z = 'z' if rot[2] != 0.0 else ''
            duration_text = '_{}_'.format(dur)
            pts_text = '{}pts_'.format(pts)
            prefix = rot_x + rot_y + rot_z + duration_text + pts_text
            return prefix


        methods = ('lmeds', 'ransac')
        #methods = [methods[0]]

        num_pts = (7, 15, 30)
        num_pts = (15, )

        duration = {'short': 0.2, 'long': 2}
        #duration = {'long': 2}
        dt = 1e-2
        steps = duration['long'] / dt
        controls = ((0., 0., 0.),
                     (0., 0., 20./steps),
                     (7./steps, 0., 20./steps),
                     (7./steps, 5./steps, 20./steps))
        #controls = [controls[3]]


        calibration_noise_levels = ('low', 'medium', 'high')

        elevation_range = ('small', 'large')
        #elevation_range = [elevation_range[1]]

        resolution_scaler = (0.5, 1)
        #resolution_scaler = (1, )

        initial_rot = self._uav.euler_rotation_zyx
        initial_coords = self._uav.coords

        for method in methods:
            for z_range in elevation_range:
                for dur in duration:
                    for pts in num_pts:
                        self._samples_per_frame = pts
                        for ctrl in controls:
                            control_duration = duration[dur]
                            control_input = {'droll_dt': ctrl[0],
                                             'dpitch_dt': ctrl[1],
                                             'dyaw_dt': ctrl[2],
                                             'duration': control_duration,
                                             'dt': dt,
                                             'radians': False}
                            for calibration_noise in calibration_noise_levels:
                                self._uav.setPose(rotation=initial_rot, coords=initial_coords)
                                print('running: {}:{}:{}:{}pts:{}:{}'.format(method, z_range, dur, pts, ctrl, calibration_noise))
                                errs_angles, errs_coords, failures, text, fname = \
                                    self.estimateEMatrixErrorDistribution(control_input=control_input,
                                                                          calibration_error_level=calibration_noise,
                                                                          elevation_range=z_range,
                                                                          method=method,
                                                                          reject_outliers=reject_outliers)
                                fname = fname_prefix(ctrl, dur, pts) + fname

                                self.plotEMatrixErrorDistribution(errs_angles, errs_coords, failures, text,
                                                                  os.path.join(save_path, fname))


                            # resolution tests
                            for s in resolution_scaler:
                                self._uav.setPose(rotation=initial_rot, coords=initial_coords)
                                print('running: {}:{}:{}:{}pts:{}:{}'.format(method, z_range, dur, pts, ctrl, s))
                                errs_angles, errs_coords, failures, text, fname = \
                                    self.estimateEMatrixErrorDistribution(control_input=control_input,
                                                                          calibration_error_level='zero',
                                                                          elevation_range=z_range,
                                                                          resolution_scaler=s,
                                                                          method=method,
                                                                          reject_outliers=reject_outliers)

                                fname = fname_prefix(ctrl, dur, pts) + fname

                                self.plotEMatrixErrorDistribution(errs_angles, errs_coords, failures, text,
                                                                  os.path.join(save_path, fname))





save_path = 'C:\\Users\\efeoz\\Desktop\\simulation_result'
cam_args = {'focal_length': 43e-3,
            'resolution': (4008, 2672),
            'pixel_size': (9e-6, 9e-6),
            'perfect_resolution': False
            }

uav_args = {'coords': (3506, 567, 5e3),
            'rotation': (np.pi, 0.0, 0.0),
            'camera_rotation': (0.0, 0.0, np.pi/2), # (roll(x), pitch(y), yaw(z)) in the order zyx wrt plane origin
            'camera_coords': (3.5, 0.0, 0.5) # (x,y,z) translation wrt plane origin
            }

sim = Simulator(uav_args=uav_args, cam_args=cam_args, samples_per_frame=15)#,
                #calibration_errors={'low': 0.05 , 'medium': 0.1, 'high': .5})

duration = 2.
dt = 1e-2
steps = 2. / dt
controls = ((0., 0., 0.),
            (0., 0., 20./steps),
            (7./steps, 0., 20./steps),
            (7./steps, 5./steps, 20./steps))

initial_rot = sim._uav.euler_rotation_zyx
initial_coords = sim._uav.coords
sim._uav.setPose(rotation=initial_rot, coords=initial_coords)


ctrl = controls[2]
control_input = {'droll_dt': ctrl[0],
                 'dpitch_dt': ctrl[1],
                 'dyaw_dt': ctrl[2],
                 'duration': 2,
                 'dt': dt,
                 'radians': False}

errs_angles, errs_coords, failures, text, fname = \
    sim.estimateEMatrixErrorDistribution(control_input=control_input,
                                         calibration_error_level='zero',
                                         elevation_range='large',
                                         method='lmeds',
                                         reject_outliers=True)
fname = 'debug_' + fname

sim.plotEMatrixErrorDistribution(errs_angles, errs_coords, failures, text,
                                  os.path.join(save_path, fname))

high_errs = np.sum(np.abs(np.degrees(errs_angles)) > 1e-3, axis=-1).astype(bool)

print('total errors: {}'.format(np.sum(high_errs)))

relative_rotation, _, cam2cam_translation = sim._uav.applyControl(**control_input)

angle_pcnt = np.abs(errs_angles)/np.abs(relative_rotation + 1e-32).reshape(1,3)
trans_pcnt = np.abs(errs_coords)/np.abs(cam2cam_translation + 1e-32).reshape(1,3)
all_errs = np.concatenate([angle_pcnt, trans_pcnt], axis=1)

print('no rotation:')
print('error% < 1%: {}'.format(np.sum(np.sum(all_errs < 0.01, axis=-1).astype(bool))))
print('error% < 5%: {}'.format(np.sum(np.sum(all_errs < 0.05, axis=-1).astype(bool))))
print('error% < 10%: {}'.format(np.sum(np.sum(all_errs < 0.1, axis=-1).astype(bool))))
print('error% < 20%: {}'.format(np.sum(np.sum(all_errs < 0.2, axis=-1).astype(bool))))
print('error% < 50%: {}'.format(np.sum(np.sum(all_errs < 0.5, axis=-1).astype(bool))))
print('errors in angles and coordinates:\n{}'.format(
    np.round(
        np.concatenate([np.degrees(errs_angles), errs_coords], axis=1) * 100) / 100)
    )



#sim.runEMatrixSimulations(save_path=save_path, reject_outliers=True)
#sim.plotEMatrixComputationTime()

