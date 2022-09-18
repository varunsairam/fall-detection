#include<algorithm>
#include <cmath>

// IMU imports
#include "I2Cdev.h"
#include "MPU6050.h"
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif

// Storage System imports
#include "FS.h"
#include "SD.h"
#include <SPI.h>
#include <WiFi.h>
#include "NTPClient_t.h"
#include <WiFiUdp.h>

// HTTP import for sending emergency message
#include <HTTPClient.h>

// ML model import
#include "FallDetectionModel.h"


// VARIABLES ------------------------------------------------------------------------------------

# define SERIAL_RATE 38400
# define WINDOW_SIZE 200 // Window size: No of readings to process for the Time Series Classifier

// IMU variables
MPU6050 accelgyro;
int16_t ax, ay, az;
int16_t gx, gy, gz;

// IMU markers mark the pins of the IMU
int IMU_markers[] = {33, 32};
int imu_nos = sizeof(IMU_markers)/sizeof(int);

// HW settings
#define LED_PIN 2
bool blinkState = false;
unsigned long previousTime = 0;

// Classifier Model Variables
float raw[WINDOW_SIZE][6];
float *agg;
int iter = 0;
Eloquent::ML::Port::LogisticRegression clf; // Model
int prediction = 0;

// SD Card Variables
String formattedDateTime;
String dataString;

// Network Settings: THE ESP32 will connect to this network on startup
const char* ssid     = "EnterYourWifi";
const char* password = "******";

// Define CS pin for the SD card module
#define SD_CS 5
String dataMessage;

// NTP Client to get time
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP);
int SD_success = 1;
String fileName;

// Message for sending the data
String url;    
bool startTimer = false; 
long startTime = 0;

// Functions --------------------------------------------

// stringTogether joins values comma delimited for logging into a csv file 
String stringTogether(int a, int b, int c, int d, int e, int f){
    return String(a) + ',' + String(b) + ',' + String(c) + ',' + String(d) + ',' + 
    String(e) + ',' + String(f);
}

String stringTogether(float a, float b, float c, float d, float e, float f, float g, float h, int pred){
    return String(a) + ',' + String(b) + ',' + String(c) + ',' + String(d) + ',' + 
    String(e) + ',' + String(f) + ',' + String(g) + ',' + String(h)+ ',' + String(pred);
}

String stringTogether(float a, float b, float c, float d, float e, float f){
    return String(a) + ',' + String(b) + ',' + String(c) + ',' + String(d) + ',' + 
    String(e) + ',' + String(f);
}


// IMU Functions

void calibrate_IMU (MPU6050 &mpu, int iter) {
    // Calibrates each IMU
    delay(3000);

    mpu.setXAccelOffset(0);
    mpu.setYAccelOffset(0);
    mpu.setZAccelOffset(0);

    mpu.setXGyroOffset(0);
    mpu.setYGyroOffset(0);
    mpu.setZGyroOffset(0);
    
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    int ax_mean = ax, ay_mean = ay, az_mean = az, gx_mean = gx, gy_mean = gy, gz_mean = gz;
    int ax_offset, ay_offset, az_offset, gx_offset, gy_offset, gz_offset;

    for (int i=0; i<iter; i++){
        
        delay(10);
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    
        ax_mean = (ax_mean+ax)/2;
        ay_mean = (ay_mean+ay)/2;
        az_mean = (az_mean+az)/2;
        gx_mean = (gx_mean+gx)/2;
        gy_mean = (gy_mean+gy)/2;
        gz_mean = (gz_mean+gz)/2;
    }

    ax_offset=-ax_mean/8;
    az_offset=-az_mean/8;
    ay_offset=(4096-ay_mean)/8;

    gx_offset=-gx_mean/4;
    gy_offset=-gy_mean/4;
    gz_offset=-gz_mean/4;

    mpu.setXAccelOffset(ax_offset);
    mpu.setYAccelOffset(ay_offset);
    mpu.setZAccelOffset(az_offset);

    mpu.setXGyroOffset(gx_offset);
    mpu.setYGyroOffset(gy_offset);
    mpu.setZGyroOffset(gz_offset);

}

void IMU_setup() {
    // Initializing the IMU (MPU6050)

    // join I2C bus (I2Cdev library doesn't do this automatically)
    #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
        Wire.begin();
    #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
        Fastwire::setup(400, true);
    #endif

    // initialize device
    Serial.println("Initializing I2C devices...");
     for (int i = 0; i < imu_nos; i++) {
        pinMode(IMU_markers[i], OUTPUT);  // defines pins as output pins
        digitalWrite(IMU_markers[i], HIGH);   // park the sensor at 0x69
    }

    for (int i = 0; i < imu_nos; i++) {
        digitalWrite(IMU_markers[i], LOW);    // set this sensor at 0x68
        delay(100);
        accelgyro.initialize();  //initializes MPU6050 sensor at 0x68
        delay(100);
        accelgyro.setFullScaleAccelRange(MPU6050_ACCEL_FS_8);
        delay(100);
        digitalWrite(IMU_markers[i], HIGH);   // park this sensor at 0x69
    }

    // verify connection
    Serial.println("Testing device connections...");
    
    for (int i = 0; i < imu_nos; i++) {
        digitalWrite(IMU_markers[i], LOW);    // set this sensor at 0x68
        delay(100);
        Serial.println(accelgyro.testConnection() ? "MPU6050 connection successful" : "MPU6050 connection failed");
        delay(100);
        digitalWrite(IMU_markers[i], HIGH);   // park this sensor at 0x69
    }
    pinMode(LED_PIN, OUTPUT);
}


void blink_for(int blinkMilliSeconds){
    // A function to blink LED for given no of milliseconds (without blocking the thread)

    unsigned long currentTime = millis();

     if (currentTime - previousTime >= blinkMilliSeconds) {
        // save the last time you blinked the LED
        previousTime = currentTime;
        blinkState = !blinkState;

        // set the LED with the ledState of the variable:
        digitalWrite(LED_PIN, blinkState);
    } 
}


float* featureExtract(float raw[WINDOW_SIZE][6]){
    // Extracting best features from raw time series data

    static float agg[8];

    // raw[0] -> ax  |   raw[1] -> ay  |   raw[2] -> az
    // raw[3] -> gx  |   raw[4] -> gy  |   raw[5] -> gz    

    float A_net[WINDOW_SIZE], G_net[WINDOW_SIZE], Ax_abs[WINDOW_SIZE], Ay_abs[WINDOW_SIZE];
    double C13_acc_area = 0; 
    float gsum1=0, gsum2=0;
    float sum_ax=0, sum_az=0, std_ax=0, std_az=0, sum_gx=0, sum_gz=0, std_gx=0, std_gz=0;
    
    // Iterating over the time series windows for certain features
    for (int i = 0; i<WINDOW_SIZE; i++){
        
        // Obataing the absolute value for accelarations along x and y at each instant
        Ax_abs[i] = abs(raw[i][0]);
        Ay_abs[i] = abs(raw[i][1]);
        
        // Peak acceleration at each instant
        A_net[i] = sqrt( (raw[i][0]*raw[i][0]) + (raw[i][1]*raw[i][1]) + (raw[i][2]*raw[i][2]) );
        
        //Obtaining aggregate sum of acceleration (later used for finding the mean)
        sum_ax += raw[i][0];
        sum_az += raw[i][2];

        // Acceleration area along x and z (this term was found to be a useful to determnine falls)        
        C13_acc_area += sqrt( (raw[i][0]*raw[i][0]) + (raw[i][2]*raw[i][2]) );

        //Peak angular velocity at each instant
        G_net[i] = sqrt( (raw[i][3]*raw[i][3]) + (raw[i][4]*raw[i][4]) + (raw[i][5]*raw[i][5]) );

        //Obtaining aggregate sum of angular velocities (later used for finding the mean)
        sum_gx += raw[i][3];
        sum_gz += raw[i][5];

        //Gravity acceleration sum at start and end of window
        if (i<15)
            gsum1 += raw[i][1];
        else if (i>=(WINDOW_SIZE-15))
            gsum2 += raw[i][1];
    }

    // Obtaining the mean values from the sum
    float mean_ax = sum_ax/WINDOW_SIZE, mean_az = sum_az/WINDOW_SIZE;
    float mean_gx = sum_gx/WINDOW_SIZE, mean_gz = sum_gz/WINDOW_SIZE;

    // Obtaining std deviations after obtaining the mean values
    for (int i = 0; i<WINDOW_SIZE; i++){
        std_ax += (raw[i][0] - mean_ax)*(raw[i][0] - mean_ax);
        std_az += (raw[i][2] - mean_az)*(raw[i][2] - mean_az);

        std_gx += (raw[i][3] - mean_gx)*(raw[i][3] - mean_gx);
        std_gz += (raw[i][5] - mean_gz)*(raw[i][5] - mean_gz);

    }
    std_ax = std_ax/WINDOW_SIZE;
    std_az = std_az/WINDOW_SIZE;

    std_gx = std_gx/WINDOW_SIZE;
    std_gz = std_gz/WINDOW_SIZE;


    // Feature 1: Max acceleration along x direction
    agg[0] = *std::max_element(Ax_abs, Ax_abs + WINDOW_SIZE); 
    
    // Feature 2: Max acceleration along y direction
    agg[1] = *std::max_element(Ay_abs, Ay_abs + WINDOW_SIZE); 

    // Feature 3: Max peak to peak total accelaration
    agg[2] = *std::max_element(A_net, A_net+WINDOW_SIZE) - *std::min_element(A_net, A_net+WINDOW_SIZE);

    // Feature 4: Standard deviation of acceleration in the horizontal plane
    agg[3] = sqrt( std_ax + std_az );

    // Feature 5: Activity signal area (horizontal plane)
    agg[4] = C13_acc_area * 0.005;

    // Feature 6: Max peak to peak total angular velocity
    agg[5] = *std::max_element(G_net, G_net+WINDOW_SIZE) - *std::min_element(G_net, G_net+WINDOW_SIZE);
    
    // Feature 7: Standard deviation of angular velocity in the horizontal plane
    agg[6] = sqrt( std_gx + std_gz );

    // Feature 8: Change in Accleration along gravity between start and end
    agg[7] = (gsum2 - gsum1)/15;


    // Scaling the terms to be centered at zero - values obtained from the data analysis
    float scales[8] = { 0.911,   0.86 ,  1.345,  0.17 ,  1.443,  116.66 ,    19.388,     0.374};
    float means[8] = {  0.65,     1.42,   1.23,   0.24,   2.29,   120.7,      26.9,       0.096};
    
    for(int i = 0; i<8; i++){
        agg[i] = (agg[i]-means[i])/scales[i];
    }
    return agg;
}


// HTTP Client Functions
void emergency_message()
{
    // Call HTTP url for emergency message
    url = "http://********";
    postData(); // calling postData to run the above-generated url once so that you will receive a message.
}

void postData()     
{
    int httpCode;     // HTTP reponse code variable
    HTTPClient http;  // HTTP Client Object
    http.begin(url);  // start HTTPClient with url
    httpCode = http.POST(url);  // Post url and get return value
    if (httpCode == 200){       // Check if the responce http code is 200
        Serial.println("Sent ok."); 
    }
    else{ 
        Serial.println("Error."); 
    }
    http.end();          // End HTTP Client upon completion
}


// SD Storage Functions
void writeFile(fs::FS &fs, const char * path, const char * message) {
    Serial.printf("Writing file: %s\n", path);
    File file = fs.open(path, FILE_WRITE);
    if(!file) {
        Serial.println("Failed to open file for writing");
        return;
    }
    if(file.print(message)) {
        Serial.println("File written");
    } else {
        Serial.println("Write failed");
    }
    file.close();
}

void appendFile(fs::FS &fs, const char * path, const char * message) {
    // Append data to the SD card
    File file = fs.open(path, FILE_APPEND);
    if(!file) {
        Serial.println("Failed to open file for appending");
        return;
    }
    if(file.print(message)) {
        // Serial.println("Message appended");
    } else {
        Serial.println("Append failed");
    }
    file.close();
}


void HW_setup() {

    // Connect to Wifi
    Serial.print("Connecting to ");
    Serial.println(ssid);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected.");
    
    // Initialize a NTPClient to get time through the internet
    timeClient.begin();
    timeClient.setTimeOffset(-21600); // Based on time zone
    
    // Initialize SD card
    SD.begin(SD_CS);  
    if(!SD.begin(SD_CS)) {
        Serial.println("Card Mount Failed");
        SD_success = 0;
        return;
    }
    uint8_t cardType = SD.cardType();
    if(cardType == CARD_NONE) {
        Serial.println("No SD card attached");
        SD_success = 0;
        return;
    }
    Serial.println("Initializing SD card...");
    if (!SD.begin(SD_CS)) {
        Serial.println("ERROR - SD card initialization failed!");
        SD_success = 0;
        return;    // init failed
    }

    // Create a file with timestamp in the name
    while(!timeClient.update()) {
        timeClient.forceUpdate();
    }
    fileName = '/' + timeClient.getFormattedDate() + ".csv";
    Serial.println("Filename:  " + fileName);

    writeFile(SD, fileName.c_str(), "Time, Ax1, Ay1, Az1, Gx1, Gy1, Gz1, Ax2, Ay2, Az2, Gx2, Gy2, Gz2, F1, F2, F3, F4, F5, F6, F7, F8, Pred \r\n");

}

// Main Setup and Loop functions ----------------------------------------------------------------

void setup() {
    Serial.begin(SERIAL_RATE);
    HW_setup(); 
    IMU_setup(); 
}


void loop() {

    // Check SD card for data logging
    if (SD_success == 0){
        Serial.println("Card Module Failed");
        return;
    }

    dataString = String(millis()) + ',';

    // read raw accel/gyro measurements from device
    for (int i = 0; i < imu_nos; i++) {
        digitalWrite(IMU_markers[i], LOW);    // set this sensor at 0x68        
        accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);  //calls values from sensor at 0x68
        dataString += stringTogether(ax,ay,az,gx,gy,gz)+',' ;

        digitalWrite(IMU_markers[i], HIGH);   // park this sensor at 0x69 (to get data from other sensor in next iteration)
    }
    
    // Converting to SI units
    raw[iter][0] = ax/4096.0;   raw[iter][1] = -ay/4096.0;      raw[iter][2] = -az/4096.0;
    raw[iter][3] = gx/131.0;    raw[iter][4] = -gy/131.0;       raw[iter][5] = -gz/131.0;
    
    
    // Iterating across time series data
    iter++;

    if (iter == WINDOW_SIZE){ // When iteration reaches window size
        
        // Obtaining features and making prediction
        agg = featureExtract(raw);
        prediction = clf.predict(agg);
        Serial.println("Prediction: "+ String(prediction));
        
        memmove(raw[0], raw[WINDOW_SIZE/2], sizeof(raw)/2); // Remove values from first half of window to make space for the next
        
        iter = WINDOW_SIZE/2; // Restart iteration from middle of current window to obain the next window
    
    }

    dataString += + "\r\n";
    Serial.print(dataString); 

    // Write data stream with prediction to SD card for logging
    appendFile(SD, fileName.c_str(), dataString.c_str());


    if (prediction == 1){
        // if fall is detected, blink rapidly
        blink_for(100);
        
        if (!startTimer)
            startTime = millis();
        startTimer = true;
        
    }
    else 
        blink_for(2000);


    if (startTimer)
        if ((millis() - startTime) > 5000)
            // Wait 5 seconds for user to abort message in case of false alarm
            // TODO: detect button press other cue to cancel emergency message
            emergency_message();
}


