/*
 * Sphere Pong Wars for M5Stack StopWatch
 *
 * Based on 3D Sphere Pong Wars:
 * https://github.com/K-Yama2010/3D_Sphere_Pong_Wars
 * Copyright (c) 2025 K-Yama2010
 * Released under the MIT License. See LICENSE.
 */

#include <Arduino.h>
#include <M5Unified.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {

constexpr int kGridSize = 8;
constexpr int kAgentPairs = 3;
constexpr int kAgentCount = kAgentPairs * 2;
constexpr float kBallSpeed = 0.085f;
constexpr float kBallSize = 0.12f;
constexpr float kReflectionRandomness = 0.15f;
constexpr float kCameraZ = 3.0f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTau = kPi * 2.0f;
constexpr uint8_t kTeamNeon = 0;
constexpr uint8_t kTeamTransparent = 1;

struct Vec2 {
  float x = 0;
  float y = 0;
};

struct Vec3 {
  float x = 0;
  float y = 0;
  float z = 0;
};

struct Quad {
  uint16_t vertex[4] = {};
  uint8_t team = 0;
  Vec3 center;
  float depth = 0;
};

struct Agent {
  uint8_t team = 0;
  Vec3 position;
  Vec3 velocity;
  int16_t currentQuad = -1;
};

M5Canvas canvas(&M5.Display);
std::vector<Vec3> vertices;
std::vector<Vec3> rotatedVertices;
std::vector<Vec2> projectedVertices;
std::vector<Quad> quads;
std::vector<uint16_t> quadOrder;
std::array<Agent, kAgentCount> agents;

int16_t screenWidth = 0;
int16_t screenHeight = 0;
float centerX = 0;
float centerY = 0;
float sphereRadius = 0;
float fov = 0;
float rotationX = 0;
float rotationY = 0;
float spinX = 0;
float spinY = 0.015f;
uint8_t colorIndex = 0;
uint32_t rngState = 1;

constexpr uint32_t kNeonColors[6] = {
  0x00FF9D, 0x00E5FF, 0xFF2ED1, 0xFFD60A, 0x9DFF00, 0xFF5A36
};

uint32_t nextRandom() {
  uint32_t value = rngState;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  rngState = value ? value : 0x6D2B79F5u;
  return rngState;
}

float randomFloat(float low, float high) {
  const float unit = (nextRandom() & 0x00FFFFFFu) / 16777215.0f;
  return low + (high - low) * unit;
}

float dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
  return { a.y * b.z - a.z * b.y,
           a.z * b.x - a.x * b.z,
           a.x * b.y - a.y * b.x };
}

float normalize(Vec3& value) {
  const float magnitude = sqrtf(dot(value, value));
  if (magnitude > 0.000001f) {
    value.x /= magnitude;
    value.y /= magnitude;
    value.z /= magnitude;
  }
  return magnitude;
}

Vec3 add(const Vec3& a, const Vec3& b) {
  return { a.x + b.x, a.y + b.y, a.z + b.z };
}

Vec3 subtract(const Vec3& a, const Vec3& b) {
  return { a.x - b.x, a.y - b.y, a.z - b.z };
}

Vec3 multiply(const Vec3& value, float scale) {
  return { value.x * scale, value.y * scale, value.z * scale };
}

Vec3 rotateView(const Vec3& source) {
  const float sinX = sinf(rotationX);
  const float cosX = cosf(rotationX);
  const float sinY = sinf(rotationY);
  const float cosY = cosf(rotationY);
  const float y1 = source.y * cosX - source.z * sinX;
  const float z1 = source.y * sinX + source.z * cosX;
  return { source.x * cosY - z1 * sinY,
           y1,
           source.x * sinY + z1 * cosY };
}

uint32_t darken(uint32_t color, float amount) {
  const uint8_t r = ((color >> 16) & 255) * amount;
  const uint8_t g = ((color >> 8) & 255) * amount;
  const uint8_t b = (color & 255) * amount;
  return (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) | b;
}

float triangleSign(const Vec2& p1, const Vec2& p2, const Vec2& p3) {
  return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

bool pointInTriangle(const Vec2& p, const Vec2& a, const Vec2& b,
                     const Vec2& c) {
  const float d1 = triangleSign(p, a, b);
  const float d2 = triangleSign(p, b, c);
  const float d3 = triangleSign(p, c, a);
  return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
}

void createQuadSphere() {
  vertices.clear();
  quads.clear();
  const Vec3 normals[6] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
  const Vec3 tangents[6] = { { 0, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 } };
  int vertexOffset = 0;
  for (int face = 0; face < 6; ++face) {
    const Vec3 bitangent = cross(normals[face], tangents[face]);
    for (int y = 0; y <= kGridSize; ++y) {
      for (int x = 0; x <= kGridSize; ++x) {
        const float u = x / static_cast<float>(kGridSize) * 2.0f - 1.0f;
        const float v = y / static_cast<float>(kGridSize) * 2.0f - 1.0f;
        Vec3 point = add(normals[face],
                         add(multiply(tangents[face], u),
                             multiply(bitangent, v)));
        normalize(point);
        vertices.push_back(point);
      }
    }
    for (int y = 0; y < kGridSize; ++y) {
      for (int x = 0; x < kGridSize; ++x) {
        const int first = vertexOffset + y * (kGridSize + 1) + x;
        const int second = first + kGridSize + 1;
        Quad quad;
        quad.vertex[0] = first;
        quad.vertex[1] = first + 1;
        quad.vertex[2] = second + 1;
        quad.vertex[3] = second;
        quad.team = face < 3 ? kTeamNeon : kTeamTransparent;
        quad.center = multiply(
          add(add(vertices[quad.vertex[0]], vertices[quad.vertex[1]]),
              add(vertices[quad.vertex[2]], vertices[quad.vertex[3]])),
          0.25f);
        quads.push_back(quad);
      }
    }
    vertexOffset += (kGridSize + 1) * (kGridSize + 1);
  }
  rotatedVertices.resize(vertices.size());
  projectedVertices.resize(vertices.size());
  quadOrder.resize(quads.size());
}

int findClosestQuad(const Vec3& position) {
  float nearestDistance = 1e9f;
  int nearest = -1;
  for (size_t i = 0; i < quads.size(); ++i) {
    const Vec3 difference = subtract(position, quads[i].center);
    const float distance = dot(difference, difference);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = i;
    }
  }
  return nearest;
}

void resetGame() {
  rngState = esp_random();
  if (!rngState) rngState = millis() | 1;
  rotationX = randomFloat(-0.20f, 0.20f);
  rotationY = randomFloat(0.0f, kTau);
  spinX = 0;
  spinY = 0.015f;
  for (size_t i = 0; i < quads.size(); ++i) {
    const int face = i / (kGridSize * kGridSize);
    quads[i].team = face < 3 ? kTeamNeon : kTeamTransparent;
  }

  for (int i = 0; i < kAgentCount; ++i) {
    Agent& agent = agents[i];
    agent.team = (i & 1) == 0 ? kTeamTransparent : kTeamNeon;
    Vec3 position = { 0, 0, 1 };
    int quad = -1;
    for (int attempt = 0; attempt < 512; ++attempt) {
      const float azimuth = randomFloat(0.0f, kTau);
      const float z = randomFloat(-0.92f, 0.92f);
      const float radial = sqrtf(1.0f - z * z);
      position = { cosf(azimuth) * radial, sinf(azimuth) * radial, z };
      if (rotateView(position).z > -0.18f) continue;
      quad = findClosestQuad(position);
      if (quad < 0 || quads[quad].team != agent.team) continue;
      bool separated = true;
      for (int previous = 0; previous < i; ++previous) {
        if (fabsf(dot(position, agents[previous].position)) > 0.78f) {
          separated = false;
          break;
        }
      }
      if (separated) break;
      quad = -1;
    }
    agent.position = position;
    agent.currentQuad = quad >= 0 ? quad : findClosestQuad(position);

    Vec3 tangent;
    do {
      tangent = { randomFloat(-1.0f, 1.0f), randomFloat(-1.0f, 1.0f),
                  randomFloat(-1.0f, 1.0f) };
      tangent = subtract(tangent, multiply(position, dot(tangent, position)));
    } while (normalize(tangent) < 0.10f);
    agent.velocity = multiply(tangent, kBallSpeed);
  }
}

void updateAgents() {
  for (Agent& agent : agents) {
    Vec3 next = add(agent.position, agent.velocity);
    normalize(next);
    Vec3 tangentVelocity = subtract(next, agent.position);
    if (normalize(tangentVelocity) > 0.000001f) {
      agent.velocity = multiply(tangentVelocity, kBallSpeed);
    }
    const int nextQuad = findClosestQuad(next);
    if (nextQuad >= 0 && agent.currentQuad >= 0 && quads[nextQuad].team != agent.team) {
      quads[nextQuad].team = agent.team;
      Vec3 wall = subtract(quads[nextQuad].center,
                           quads[agent.currentQuad].center);
      normalize(wall);
      wall.x += randomFloat(-kReflectionRandomness, kReflectionRandomness);
      wall.y += randomFloat(-kReflectionRandomness, kReflectionRandomness);
      wall.z += randomFloat(-kReflectionRandomness, kReflectionRandomness);
      normalize(wall);
      agent.velocity = subtract(
        agent.velocity, multiply(wall, 2.0f * dot(agent.velocity, wall)));
    }
    agent.position = add(agent.position, agent.velocity);
    normalize(agent.position);
    agent.currentQuad = findClosestQuad(agent.position);
  }
}

void updateTouchRotation() {
  static bool touching = false;
  static int16_t previousX = 0;
  static int16_t previousY = 0;
  if (M5.Touch.getCount()) {
    const auto detail = M5.Touch.getDetail(0);
    if (detail.isPressed() || detail.wasPressed()) {
      if (touching) {
        spinY += (detail.x - previousX) * 0.00085f;
        spinX += (detail.y - previousY) * 0.00085f;
        spinX = std::max(-0.11f, std::min(0.11f, spinX));
        spinY = std::max(-0.11f, std::min(0.11f, spinY));
      }
      previousX = detail.x;
      previousY = detail.y;
      touching = true;
    }
  } else {
    touching = false;
  }
  rotationX += spinX;
  rotationY += spinY;
  spinX *= 0.994f;
  spinY = 0.015f + (spinY - 0.015f) * 0.994f;
}

void drawScene() {
  const uint32_t neon = kNeonColors[colorIndex];
  const uint32_t darkNeon = darken(neon, 0.29f);
  canvas.fillSprite(TFT_BLACK);

  for (size_t i = 0; i < vertices.size(); ++i) {
    rotatedVertices[i] = rotateView(vertices[i]);
    const float depth = rotatedVertices[i].z + kCameraZ;
    const float scale = fov / depth;
    projectedVertices[i] = { centerX + rotatedVertices[i].x * scale,
                             centerY + rotatedVertices[i].y * scale };
  }
  for (size_t i = 0; i < quads.size(); ++i) {
    quads[i].depth = rotateView(quads[i].center).z;
    quadOrder[i] = i;
  }
  std::sort(quadOrder.begin(), quadOrder.end(), [](uint16_t a, uint16_t b) {
    return quads[a].depth > quads[b].depth;
  });

  std::vector<std::array<Vec2, 4>> visibleFrontNeon;
  visibleFrontNeon.reserve(quads.size() / 2);
  for (uint16_t index : quadOrder) {
    const Quad& quad = quads[index];
    if (quad.team == kTeamTransparent) continue;
    const Vec3 edge1 = subtract(rotatedVertices[quad.vertex[1]],
                                rotatedVertices[quad.vertex[0]]);
    const Vec3 edge2 = subtract(rotatedVertices[quad.vertex[2]],
                                rotatedVertices[quad.vertex[0]]);
    const Vec3 normal = cross(edge1, edge2);
    //const float facing = dot(normal, multiply(rotatedVertices[quad.vertex[0]], -1));
    const Vec3& surfacePoint = rotatedVertices[quad.vertex[0]];
    const Vec3 surfaceToCamera = {
      -surfacePoint.x, -surfacePoint.y, -kCameraZ - surfacePoint.z
    };
    const float facing = dot(normal, surfaceToCamera);
    std::array<Vec2, 4> points = {
      projectedVertices[quad.vertex[0]], projectedVertices[quad.vertex[1]],
      projectedVertices[quad.vertex[2]], projectedVertices[quad.vertex[3]]
    };
    if (facing >= 0) visibleFrontNeon.push_back(points);
    const uint32_t faceColor = facing < 0 ? darkNeon : neon;
    canvas.fillTriangle(points[0].x, points[0].y, points[1].x, points[1].y,
                        points[2].x, points[2].y, faceColor);
    canvas.fillTriangle(points[0].x, points[0].y, points[2].x, points[2].y,
                        points[3].x, points[3].y, faceColor);
  }

  for (const Agent& agent : agents) {
    const Vec3 rotatedPosition = rotateView(agent.position);
    const bool onBack = rotatedPosition.z > 0;
    const float depth = rotatedPosition.z + kCameraZ;
    const float scale = fov / depth;
    const Vec2 center = { centerX + rotatedPosition.x * scale,
                          centerY + rotatedPosition.y * scale };
    bool occluded = false;
    if (onBack) {
      for (const auto& quad : visibleFrontNeon) {
        if (pointInTriangle(center, quad[0], quad[1], quad[2]) || pointInTriangle(center, quad[0], quad[2], quad[3])) {
          occluded = true;
          break;
        }
      }
    }
    if (onBack && occluded) continue;

    uint32_t ballColor;
    if (agent.team == kTeamNeon) {
      const int behind = findClosestQuad(multiply(agent.position, -1));
      ballColor = behind >= 0 && quads[behind].team == kTeamNeon
                    ? darkNeon
                    : TFT_BLACK;
    } else {
      ballColor = onBack ? darkNeon : neon;
    }

    Vec3 normal = agent.position;
    Vec3 up = fabsf(normal.y) > 0.9f ? Vec3{ 1, 0, 0 } : Vec3{ 0, 1, 0 };
    Vec3 tangent = cross(up, normal);
    normalize(tangent);
    Vec3 bitangent = cross(normal, tangent);
    normalize(bitangent);
    const float halfSize = kBallSize / depth;
    Vec3 corners[4] = {
      add(agent.position, add(multiply(tangent, -halfSize), multiply(bitangent, halfSize))),
      add(agent.position, add(multiply(tangent, halfSize), multiply(bitangent, halfSize))),
      add(agent.position, add(multiply(tangent, halfSize), multiply(bitangent, -halfSize))),
      add(agent.position, add(multiply(tangent, -halfSize), multiply(bitangent, -halfSize)))
    };
    Vec2 points[4];
    for (int i = 0; i < 4; ++i) {
      normalize(corners[i]);
      const Vec3 rotated = rotateView(corners[i]);
      const float cornerScale = fov / (rotated.z + kCameraZ);
      points[i] = { centerX + rotated.x * cornerScale,
                    centerY + rotated.y * cornerScale };
    }
    canvas.fillTriangle(points[0].x, points[0].y, points[1].x, points[1].y,
                        points[2].x, points[2].y, ballColor);
    canvas.fillTriangle(points[0].x, points[0].y, points[2].x, points[2].y,
                        points[3].x, points[3].y, ballColor);
  }
  canvas.pushSprite(0, 0);
}

void reportFps() {
  static uint32_t frameCount = 0;
  static uint32_t previousReportMs = millis();
  ++frameCount;

  const uint32_t now = millis();
  const uint32_t elapsedMs = now - previousReportMs;
  if (elapsedMs >= 1000) {
    const float fps = frameCount * 1000.0f / elapsedMs;
    Serial.printf("FPS: %.1f | agents: %d\n", fps, kAgentCount);
    frameCount = 0;
    previousReportMs = now;
  }
}

}  // namespace

void setup() {
  auto config = M5.config();
  M5.begin(config);
  Serial.begin(115200);
  M5.Display.setBrightness(115);
  screenWidth = M5.Display.width();
  screenHeight = M5.Display.height();
  centerX = screenWidth * 0.5f;
  centerY = screenHeight * 0.5f;
  sphereRadius = std::min(screenWidth, screenHeight) * 0.485f;
  fov = sphereRadius * 2.0f;
  canvas.setColorDepth(16);
  canvas.setPsram(true);
  if (!canvas.createSprite(screenWidth, screenHeight)) {
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setTextColor(TFT_RED);
    M5.Display.setTextDatum(middle_center);
    M5.Display.drawString("Canvas allocation failed", centerX, centerY);
    while (true) M5.delay(1000);
  }
  createQuadSphere();
  resetGame();
}

void loop() {
  M5.update();
  if (M5.BtnA.wasClicked()) resetGame();
  if (M5.BtnB.wasClicked()) colorIndex = (colorIndex + 1) % 6;
  updateTouchRotation();
  updateAgents();
  drawScene();
  reportFps();
  M5.delay(1);
}
