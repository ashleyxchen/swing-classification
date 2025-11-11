
import processing.serial.*;
import peasy.*;

PeasyCam cam;
Serial myPort;
float roll, pitch, yaw;
boolean gotData = false;

void setup() {
  size(800, 600, P3D);
  pixelDensity(1);  // avoid Retina scaling issues
  surface.setTitle("ICM20948 Orientation Visualization");

  println("Available ports:");
  printArray(Serial.list());
  delay(2000);

  // Arduino port
  myPort = new Serial(this, Serial.list()[3], 115200);
  myPort.bufferUntil('\n');

  cam = new PeasyCam(this, 400);
  cam.setMinimumDistance(100);
  cam.setMaximumDistance(800);
}

void draw() {
  background(30);
  lights();

  if (!gotData) {
    fill(255);
    textAlign(CENTER);
    text("Waiting for IMU data...", width/2, height/2);
    return;
  }

  pushMatrix();
  //translate(width/2, height/2, 0);

  // skip NaN values?
//   if (Float.isNaN(roll) || Float.isNaN(pitch) || Float.isNaN(yaw)) {
//     popMatrix();
//     return;
//   }

  // apply rotations - order matterssss
  float speed = 2.5; // increase motion sensitivity
  rotateZ(radians(yaw * speed));
  rotateY(radians(pitch * speed));
  rotateX(radians(roll * speed));


  // draw cube
  fill(0, 255, 0);
  stroke(255);
  //box(120, 20, 40); // dimensions of box: x,y,z
  box(40, 120, 20);

  // Add axis lines
  strokeWeight(3);
  stroke(255, 0, 0); line(0, 0, 0, 100, 0, 0);   // X axis (red)
  stroke(0, 255, 0); line(0, 0, 0, 0, 100, 0);   // Y axis (green)
  stroke(0, 0, 255); line(0, 0, 0, 0, 0, 100);   // Z axis (blue)

  popMatrix();
}

void serialEvent(Serial p) {
  String line = p.readStringUntil('\n');
  if (line == null) return;

  line = trim(line);
  if (line.length() == 0) return;

  String[] tokens = splitTokens(line); // handles tabs/spaces
  if (tokens.length >= 3) {
    try {
      roll  = float(tokens[0]);
      pitch = float(tokens[1]);
      yaw   = float(tokens[2]);
      gotData = true;
      println("roll:", roll, "pitch:", pitch, "yaw:", yaw);
    } catch (Exception e) {
      println("Parse error:", line);
    }
  }
}