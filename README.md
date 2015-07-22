# Camera-to-IMU calibration toolbox
This toolbox provides a python library to perform joint calibration of a camera gyroscope system.

Given gyroscope and video data, this library can find the following parameters

* True gyroscope rate
* Time offset
* Rotation between camera and gyroscope coordinate frames
* Gyroscope measurement bias

If you use the package for your work, please cite the following paper

> Ovrén, H and Forssén, P.-E. "Gyroscope-based video stabilisation with auto-calibration." In 2015 IEEE International Conference on Robotics and Automation (ICRA) (pp. 2090–2097). Seattle, WA

## Changes from 1.0
The 2.0 version of crisp features a new fully automatic calibrator.
This means that there is no compelling reason to use the semi-manual methods in the previous version of crisp.
Therefore the old example scripts have been removed, and the old functions are not imported into the module namespace.
No old functions have been removed, so if you want to use them they are still available in submodules.

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

## Usage
The gyroscope and video data are first loaded into a stream object (`GyroStream`, and a subclass of `VideoStream` respectively).
To be able to understand how points are mapped from the real world to the image, the video stream also need a `CameraModel` (-subclass) instance.

    import crisp
    
    gyro = crisp.GyroStream.from_data(some_data_array)
    camera_model = crisp.AtanCameraModel(...) # One specific choice of camera model
    video = crisp.VideoStream.from_file(camera_model, video_file_path)


We then tie the streams together using a `AutoCalibrator` instance.
Since the calibration proces need to have estimates of the time offset and relative rotation,
these are first estimated using the `initialize()` member. This initialization only requires that
you give an approximate gyroscope sample rate (in Hz).

    calibrator = crisp.AutoCalibrator(video, gyro)
    calibrator.initialize(guessed_gyro_rate)
    result = calibrator.calibrate() # Dict of calibrated parameters

Initialization and calibration errors can be caught by handling `InitializationError` and `CalibrationError`.

### Example scripts
We bundle one example script `gopro_dataset_example.py` which shows how to use the 
library with the data in our dataset (http://www.cvl.isy.liu.se/research/datasets/gopro-gyro-dataset/).
This is the same dataset that was used to produce the above mentioned ICRA 2015 paper.

## Feedback
* For any questions regarding the method and paper, please send an e-mail to hannes.ovren@liu.se.
* For issues about the code, you are welcome to either use the tools (issue reporting, etc.) provided by GitHub, or send an e-mail.

## License
All code in this repository is licensed under the GPL version 3.
