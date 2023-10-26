# AI Music - Body Pose Detection for Controlling Music Parameters

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project harnesses the power of body pose detection to manipulate music parameters such as tempo or volume, determined by the user's movements. Utilizing the Mediapipe library, the program detects the user's body landmarks and computes their velocity over time. The resulting pose data is transformed into OSC messages which can then interface with music software such as Ableton Live or Max/MSP.

![music_pose](https://user-images.githubusercontent.com/17069785/227744014-4da1efee-03a4-4cc4-a96f-0af867840a21.png)

### What is OSC?

Open Sound Control (OSC) is a protocol for networking sound synthesizers, computers, and other multimedia devices for purposes such as musical performance or show control. More about OSC can be found [here](https://opensoundcontrol.org/).

## System Requirements

- A webcam or equivalent video capture device.
- A system with at least 8GB RAM and a quad-core processor for optimal real-time video processing.
- Compatible with Windows, macOS, and Linux operating systems.

## Installation

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To initiate the program, execute:

```bash
python main.py [--ip IP_ADDRESS] [--port PORT_NUMBER] [--no-osc]
```

**Optional arguments:**

- `--ip IP_ADDRESS`: Set the IP address for OSC messages. Default is `127.0.0.1`.
- `--port PORT_NUMBER`: Designate the port number for OSC messages. Default is `9000`.
- `--no-osc`: Option to disable sending OSC messages. Default is `False`.
- `--image`: Option to use an image instead of webcam. Default is `None`. Here the path to the image should be added.
- `--video`: Option to use a video instead of webcam. Default is `None`. Here the path to the video should be added here.

Upon execution, a video stream from the default camera will be displayed, showcasing the user's pose landmarks in real-time. If enabled, OSC messages will be transmitted to the indicated IP address and port. Exit the program by pressing the `q` key.

## Customization

For those wishing to tailor the software:

- **`pose_detector.py`**: Houses the `PoseDetector` class which oversees body pose landmark detection and pose data computations. Alter for adjustments in data detection and calculation.
  
- **`osc_sender.py`**: Contains the `OSCSender` class, responsible for OSC message transmission. Modify this to adjust how OSC messages are relayed.
  
- **`body_parts.py`**: Here lies the `BodyParts` class, defining the detected body parts by the `PoseDetector`. Adjustments can be made to manage which body parts are detected.
  
- **`util.py`**: Utility functions used across various files are found here. Add or tweak functions as required.


## Contributing

Your contributions are welcomed! Whether it's a bug report, feature suggestion, or a code contribution, every bit helps. Start by creating an issue or submitting a pull request.


## License

This project is licensed under the MIT License. Dive into the `LICENSE` file for detailed information.