# Fall Detection System

Code from parts of my effort while developing a wearable for fall detection for my Master's Project.

This effort involved using a ESP32 Microcontroller made on a custom made PCB for the data with two IMUs kept a distance apart:

![Alt text](img/setup.jpg?raw=true "Prototype")

The device and code together present a fall detection system based on Logistic Regression model running in real time on the embedded chip. The system is also capable of sending an emergency signal over HTTP through WiFi when a fall is detected.

![Alt text](img/emergency_signal.jpg?raw=true "Emergency Message and Call")



