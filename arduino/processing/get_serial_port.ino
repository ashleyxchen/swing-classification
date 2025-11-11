import processing.serial.*;
import peasy.*;

PeasyCam cam;
Serial myPort;
float roll, pitch, yaw;
boolean gotData = false;

void setup() {
 size(800, 600, P3D);
 pixelDensity(1); // disables Retina scaling warning
 printArray(Serial.list());
 delay(2000); // wait for port to free up
 myPort = new Serial(this, Serial.list()[3], 115200);  // set correct index [3] serial port-1130
 myPort.bufferUntil('\n');
 cam = new PeasyCam(this, 200);
}
