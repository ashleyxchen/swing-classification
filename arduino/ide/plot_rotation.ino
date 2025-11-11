
// SERIAL PLOTTER
#include <Wire.h>

// ICM20948 I2C Address and Registers
#define ICM20948_ADDRESS 0x68   // *******change to 0x69 if AD0 pin is high
#define ACCEL_XOUT_H 0x2D
#define GYRO_XOUT_H  0x33

void setup() {
  Serial.begin(115200);   // higher baud rate for smoother plotting
  Wire.begin();
  Wire.setClock(400000);  // optional: faster I2C -- 500kHz

  Serial.println("Starting ICM20948 initialization...");

  uint8_t whoAmI = readRegister(ICM20948_ADDRESS, 0x00); // WHO_AM_I register
  Serial.print("WHO_AM_I register: 0x");
  Serial.println(whoAmI, HEX);

  if (whoAmI != 0xEA) {
    Serial.println("ICM20948 not detected! Check connections and address.");
    while (1);
  }

  // Wake up the sensor and select clock source
  writeRegister(ICM20948_ADDRESS, 0x06, 0x01); // PWR_MGMT_1: clock source = auto select
  delay(10);

  Serial.println("ICM20948 initialized successfully!");
}

void loop() {
  float gx, gy, gz;

  readGyroscope(gx, gy, gz);

  // Print data in a plot-friendly format
  // Each line = one time step, each variable separated by tab
  Serial.print(gx); Serial.print("\t");
  Serial.print(gy); Serial.print("\t");
  Serial.println(gz);

  delay(50);  // ~20 Hz update rate for Serial Plotter
}

// ----- SENSOR FUNCTIONS -----

void readGyroscope(float &gx, float &gy, float &gz) {
  int16_t rawX = (readRegister(ICM20948_ADDRESS, GYRO_XOUT_H) << 8) | readRegister(ICM20948_ADDRESS, GYRO_XOUT_H + 1);
  int16_t rawY = (readRegister(ICM20948_ADDRESS, GYRO_XOUT_H + 2) << 8) | readRegister(ICM20948_ADDRESS, GYRO_XOUT_H + 3);
  int16_t rawZ = (readRegister(ICM20948_ADDRESS, GYRO_XOUT_H + 4) << 8) | readRegister(ICM20948_ADDRESS, GYRO_XOUT_H + 5);

  gx = rawX / 131.0; // ±250 dps range → 131 LSB/(°/s)
  gy = rawY / 131.0;
  gz = rawZ / 131.0;
}

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