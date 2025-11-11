// PROCESSING
#include <Wire.h>
#include <MadgwickAHRS.h>

// ----- ICM-20948 configuration -----
#define ICM20948_ADDRESS 0x68   // Use 0x69 if AD0 is pulled high
#define ACCEL_XOUT_H 0x2D
#define GYRO_XOUT_H  0x33

Madgwick filter;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);  // 400kHz I2C

  filter.begin(100); // Initialize Madgwick filter at 100 Hz
  initICM20948();

  Serial.println("ICM20948 + Madgwick filter ready!");
}

// ----- MAIN LOOP -----
void loop() {
  float ax, ay, az, gx, gy, gz;

  readIMU(ax, ay, az, gx, gy, gz);

  // Convert gyro from deg/s → rad/s for Madgwick
  filter.updateIMU(gx * DEG_TO_RAD, gy * DEG_TO_RAD, gz * DEG_TO_RAD, ax, ay, az);

  // Get Euler angles (in degrees)
  float roll  = filter.getRoll();
  float pitch = filter.getPitch();
  float yaw   = filter.getYaw();

  // Print for visualization
  Serial.print(roll);  Serial.print("\t");
  Serial.print(pitch); Serial.print("\t");
  Serial.println(yaw);

  delay(10); // ~100 Hz update rate
}

// ----- SENSOR SETUP -----
void initICM20948() {
  Serial.println("Initializing ICM20948...");

  uint8_t whoAmI = readRegister(ICM20948_ADDRESS, 0x00);
  Serial.print("WHO_AM_I: 0x"); Serial.println(whoAmI, HEX);

  if (whoAmI != 0xEA) {
    Serial.println("ICM20948 not detected! Check wiring/address.");
    while (1);
  }

  // Wake up the chip (clear sleep bit)
  writeRegister(ICM20948_ADDRESS, 0x06, 0x01);
  delay(10);

  Serial.println("ICM20948 initialized successfully!");
}

// ----- READ IMU -----
void readIMU(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  // Accelerometer
  int16_t rawAX = (readRegister(ICM20948_ADDRESS, ACCEL_XOUT_H) << 8) | readRegister(ICM20948_ADDRESS, ACCEL_XOUT_H + 1);
  int16_t rawAY = (readRegister(ICM20948_ADDRESS, ACCEL_XOUT_H + 2) << 8) | readRegister(ICM20948_ADDRESS, ACCEL_XOUT_H + 3);
  int16_t rawAZ = (readRegister(ICM20948_ADDRESS, ACCEL_XOUT_H + 4) << 8) | readRegister(ICM20948_ADDRESS, ACCEL_XOUT_H + 5);

  // Gyroscope
  int16_t rawGX = (readRegister(ICM20948_ADDRESS, GYRO_XOUT_H) << 8) | readRegister(ICM20948_ADDRESS, GYRO_XOUT_H + 1);
  int16_t rawGY = (readRegister(ICM20948_ADDRESS, GYRO_XOUT_H + 2) << 8) | readRegister(ICM20948_ADDRESS, GYRO_XOUT_H + 3);
  int16_t rawGZ = (readRegister(ICM20948_ADDRESS, GYRO_XOUT_H + 4) << 8) | readRegister(ICM20948_ADDRESS, GYRO_XOUT_H + 5);

  // Convert to physical units
  ax = rawAX / 16384.0; // ±2g
  ay = rawAY / 16384.0;
  az = rawAZ / 16384.0;

  gx = rawGX / 131.0;   // ±250 dps
  gy = rawGY / 131.0;
  gz = rawGZ / 131.0;
}

// ----- LOW-LEVEL I2C FUNCTIONS -----
uint8_t readRegister(uint8_t deviceAddress, uint8_t registerAddress) {
  Wire.beginTransmission(deviceAddress);
  Wire.write(registerAddress);
  Wire.endTransmission(false);

  Wire.requestFrom(deviceAddress, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

void writeRegister(uint8_t deviceAddress, uint8_t registerAddress, uint8_t value) {
  Wire.beginTransmission(deviceAddress);
  Wire.write(registerAddress);
  Wire.write(value);
  Wire.endTransmission();
}