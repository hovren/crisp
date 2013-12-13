# Camera-to-IMU calibration toolbox
This toolbox provides a python library and sample applications to perform the following things

1. Calibrate the relative pose between a gyroscope and a camera
1. Synchronize the time between a gyroscope and a camera

The library API consists of only a handful of important functions, and are designed to hopefully
be easy to include in other projects.
The example scripts can be used to synchronize and calibrate your own data, but are written mainly to show how to use the library API.

If you use the package for your work, please cite the following paper

> Hannes Ovrén, Per-Erik Forssén, David Törnqvist, "[Why Would I Want a Gyroscope on my RGB-D Sensor?](http://users.isy.liu.se/cvl/perfo/abstracts/ovren13.html)", IEEE Workshop on Robot Vision 2013, Clearwater Beach, Florida, USA, January 16-17, 2013.

## Installation
To use the package you need the following Python packages:

* NumPy
* SciPy
* OpenCV
* matplotlib

To build, you also need the Cython package.

To build and install the `crisp` module just run the following commands:

    $ python setup.py build
    $ python setup.py install
    
For a user-only installation add `--user` to the install command.

## Example scripts
We have bundled three example scripts so you can quickly get going. These are

1. crisppose - Find the relative pose (rotation) between a camera and IMU
1. crisptime - Find the time offset between camera and IMU data

They will have to be run in the order described above, since we need the pose to get a good time offset estimation.

### Before you start
To get started you need to have captured video together with IMU/gyro data.
For pose estimation, you need a special sequence which is described in detail below.

When you have collected the data, extract all frames and store in a directory (e.g. using ffmpeg).
In this directory, create a CSV file that (in order) lists all files and their timestamp

    frame_000.png, 0.0
    frame_001.png, 0.03
    frame_002.png, 0.06
    ...
    
For the gyroscope data, make sure it is scaled correctly and expressed in rad/s.
Save the gyro data as a (3, N) matrix with the name 'gyro', together with the timestamps of each sample as a (1,N) matrix/array called 'timestamps', as a Matlab-format MAT-file. You can use the `scipy.io.savemat` function for this.

Also save the camera calibration matrix in a MAT-file, as the variable 'K'.

Try to find out the *readout time* of your camera. Worst case, you can approximate it as 1 / f, where f is the camera framerate (e.g. f=30.0 frames/second). Store this in the same file as the camera calibration matrix with the variable name 'readout_time'.

The last parameter you need is the multiplicative factor between the IMU time and camera time.
Preferably, this should be calibrated by measuring a long sequence. If you have a very good estimate of the sample rate of your sensors, you could directly estimate it as `m = f_camera / f_imu`.
Note that an error in the multiplicative factor will affect the result significantly.

### crisppose - Estimating the relative pose
We estimate the relative pose by finding corresponding rotation axes in the data.
This set of corresponding axes must come from rotations around at least two non-parallel axes. The more orthogonal the better.

It is **very** important that the sensor is standing completely still before and after each rotation. Otherwise, rolling shutter effects can destroy your result.

Now call the script as

    $ crisppose /path/to/data/images.csv /path/to/data/imu_data.mat /path/to/camera_params.mat pose.mat --num 4
    
You will now be prompted to choose 4 (`--num`) corresponding sequences in the calibration sequence.
After choosing points from the first graph, select the same sequence in the second graph.
The resulting pose will be stored in pose.mat.

### crisptime - Find time offset 
If you only want a rough estimate you can simply call

    $ crisptime /path/to/data/images.csv /path/to/data/imu_data.mat 12.345
    
Here 12.345 is the multiplicative factor `m` between the camera and gyro such that `t_camera = m * t_gyro`.
    
A rough time offset is calculated by correlation (ZNCC using pyramids for speed) from all available data.

To get a refined offset that takes rolling shutter into account, you must specify camera calibration parameters (calibration matrix and readout time).
If your IMU data is not already expressed in the camera coordinate frame, you also need to give the relative pose (as a MAT-file that contains the variable 'R').

    $ crisptime /path/to/data/images.csv /path/to/data/imu_data.mat --camera-params /path/to/camera_params.mat --relative-pose /path/to/relative_pose.mat
    
By specifying the `--manual` flag, the rough time offset is calculated by correlation on only a small part of your data that you select in a plot window. You select matching sequences in the camera optical flow magnitude, and the gyroscope data.
The selected frames are also used in the refinement step, so try to choose a part without motion blur since points need to be tracked.
If the `--manual` flag is not set, we instead try to automatically find a part of the signal that is good, and use that.

## The `crisp` library
To use the library in your own application, simply `import crisp` and get going.

The `crisp` namespace contains the functions you most likely want to use.
There is also a number of sub-modules (e.g. `crisp.pose`, `crisp.imu`) which contain
functionality that you might find useful.

To get a feel for how the library works, we recommend that you look at the example scripts
as these were specifically written to be used as documentation.

## License
All code in this repository is licensed under the GPL version 3.
