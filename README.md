# Team-ETA

## Healthcare Monitoring System in an Autonomous Vehicle

A perfect autonomous vehicle is incomplete if it cannot take care of its passengers lest someone might face a medical emergency. Early detection of a health condition can result in a good treatment outcome, hence lowering the risk of serious health complications. In the case of a conventional vehicle, should such a situation arise, the driver can always take the necessary steps to ensure the wellbeing of the passenger. But, when it comes to an autonomous vehicle, this feature is missing and the health of the passengers goes for a toss if they face such a condition. To address this issue, the vehicle would contain an Automated Health Detection System to constantly keep a track of passenger health and accordingly adjust the vehicle parameters by taking passenger comfort into consideration.

The autonomous vehicle would have sensors like pulse rate monitor, blood pressure monitoring system, thermometer, oximeter, and so on. An advanced face recognition system would identify the age of the passenger and make necessary decisions accordingly. Additionally, the vehicle would also contain a holographic display where the passenger would input health-related information simply by marking checkboxes or by using a voice assistant. Based on the inputs given, results would be predicted using Machine Learning and Deep Learning techniques. Furthermore, these results would be saved on the cloud for future reference and better decision-making.

To implement this, disease and symptoms related datasets were collected and cleaned by performing preprocessing and Feature engineering techniques. Support Vector Machine, Naive Bayes, and Random Forest machine learning algorithms were used for model training. A robust model was developed by combining all three above mentioned algorithms to achieve cent percent accuracy and perfect predictions. This newly created model was trained on the entire data and then tested on the test dataset.

A function was created to take symptoms as inputs and make predictions about the disease identified. According to the output, that is, the predicted health condition, necessary changes required in the vehicle’s driving pattern to ensure passenger’s comfort during the journey were printed. The autonomous vehicle would then take action accordingly based on the output commands with the help of interaction between the vehicle’s AI and actuators.

## Equipping the Autonomous Vehicle with Human-like Reflexes

We humans are equipped with an involuntary response system that produces automatic reactions to certain kinds of stimuli. These
reactions are called reflexes. These actions are
performed without any prior processing and are
simply automatic as they occur without
conscious thought of the mind. For example, a
human reflex to having a hand touch a hot
object would be the immediate removal of the
hand from the object so as to avoid any
damage.

Self-driving vehicles have reached new heights
in recent times but yet are not able to function
completely on their own. This is because they
require a lot of computation for detecting
obstacles that could possibly pop up in the
vehicles’ path, therefore making the entire
process of detection and avoidance of an
obstacle slower. Hence, implementing the
concept of reflexes in autonomous vehicles
would be revolutionary, especially in making the
vehicles ready for situations like collisions, as it would require fewer computations and have a
faster response time.

The simulation for depicting the concept of
reflexes was performed using the Robot
Operating System (ROS). A comparison was
made between Faster R-CNN and the SSD
model, both of which are object detection
models. Faster R-CNN uses a region proposal
network to create boundary boxes and utilizes
those boxes to classify objects. The whole
process runs at around 7 frames per second,
which is far below what real-time processing
needs. Hence, this model proved to be inefficient
in the case of an autonomous vehicle. On the
other hand, a SSD model was designed for
object detection in real-time. SSD speeds up the
process significantly by eliminating the need for
the region proposal network. To recover the drop
in accuracy, it applies a few improvements
including multi-scale features and default boxes.
These improvements allow it to match the Faster
R-CNN’s accuracy using lower resolution
images, which further pushes the speed higher.
Similar to the aforementioned comparison, the SSD did achieve real-time processing speed
and even surpassed the accuracy of the Faster
R-CNN. Hence, a SSD was chosen for the
implementation of reflexes.

A simulation was performed for which OSRF’s
Citysim and MCity environment worlds were
used. The images in these environments were
used for the creation of the dataset. This dataset
was then divided into 2 parts: Training and
Testing. The SSD was then processed using the
Training group, followed by validation using the
Testing dataset. Next, the RVIZ setup was done.
This displayed information about all sensors and
cameras of the vehicle. Then, a ROS publisher
was implemented, which took inputs via the
cameras and sensors. It was inferred that in
case of an obstacle coming suddenly in the way
of the vehicle, the obstacle was immediately
detected, and the vehicle was compelled to apply brakes promptly or change its course with
minuscule amounts of internal computation as
shown in Img 1. This ensures immediate
reaction to sensor and camera data and
therefore is effective towards the implementation
of human-like reflexes on an autonomous
vehicle.


