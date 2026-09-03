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
constexpr int kAgentPairs = 4;
constexpr int kAgentCount = kAgentPairs * 2;
constexpr int kVertexCount = 6 * (kGridSize + 1) * (kGridSize + 1);
constexpr int kQuadCount = 6 * kGridSize * kGridSize;
constexpr int kCanvasSize = 336;
constexpr float kCanvasScaleReference = 336.0f;
constexpr float kBallSpeed = 0.085f;
constexpr float kBallSize = 0.12f;
constexpr float kReflectionRandomness = 0.15f;
constexpr float kCameraZ = 3.0f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTau = kPi * 2.0f;
constexpr float kAutoSpinRadiansPerSecond = 0.55f;
constexpr float kGravityFilter = 0.12f;
constexpr float kUprightBeginG = 0.35f;
constexpr float kUprightFullG = 0.60f;
constexpr float kLevelSpringPerSecond2 = 0.9f;
constexpr float kLevelDampingPerSecond = 1.5f;
constexpr float kMaxLevelRateRadiansPerSecond = 0.65f;
constexpr float kTouchRadiansPerPixel = 0.0055f;
constexpr float kTouchCoastDampingPerSecond = 0.55f;
constexpr float kTouchReturnDampingPerSecond = 1.1f;
constexpr float kTouchReturnSpringPerSecond2 = 0.55f;
constexpr uint32_t kTouchReturnDelayMs = 1000;
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

struct Quaternion {
  float w = 1;
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

constexpr Vec3 kFaceNormals[6] = {
  { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 },
  { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }
};
constexpr Vec3 kFaceTangents[6] = {
  { 0, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 },
  { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }
};
constexpr Vec3 kFaceBitangents[6] = {
  { 0, 0, 1 }, { 0, 0, -1 }, { 0, 0, -1 },
  { 0, 0, 1 }, { 0, 1, 0 }, { 0, -1, 0 }
};

M5Canvas canvas(&M5.Display);
std::vector<Vec3> vertices;
std::vector<Vec3> rotatedVertices;
std::vector<Vec2> projectedVertices;
std::vector<Quad> quads;
std::vector<uint16_t> quadOrder;
std::vector<std::array<Vec2, 4>> visibleFrontNeon;
std::array<Agent, kAgentCount> agents;

int16_t screenWidth = 0;
int16_t screenHeight = 0;
int16_t canvasWidth = 0;
int16_t canvasHeight = 0;
int16_t canvasOffsetX = 0;
int16_t canvasOffsetY = 0;
float centerX = 0;
float centerY = 0;
float sphereRadius = 0;
float fov = 0;
float autoRotationY = 0;
float levelRotationZ = 0;
float levelAngularVelocityZ = 0;
Quaternion touchOrientation;
float touchVelocityX = 0;
float touchVelocityY = 0;
float touchVelocityZ = 0;
uint32_t touchReleasedMs = 0;
float viewM00 = 1;
float viewM01 = 0;
float viewM02 = 0;
float viewM10 = 0;
float viewM11 = 1;
float viewM12 = 0;
float viewM20 = 0;
float viewM21 = 0;
float viewM22 = 1;
uint8_t colorIndex = 0;
uint32_t rngState = 1;
float filteredAccelX = 0;
float filteredAccelY = 0;
float filteredAccelZ = 1;
float frameDeltaSeconds = 1.0f / 30.0f;
bool levelInitialized = false;
uint32_t previousImuMs = 0;
uint32_t vibrationStopMs = 0;

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

bool deadlineReached(uint32_t now, uint32_t deadline) {
  return static_cast<int32_t>(now - deadline) >= 0;
}

void triggerHaptic(uint8_t level, uint16_t durationMs) {
  const uint32_t now = millis();
  M5.Power.setVibration(level);
  vibrationStopMs = now + durationMs;
}

void updateHaptic() {
  const uint32_t now = millis();
  if (vibrationStopMs && deadlineReached(now, vibrationStopMs)) {
    M5.Power.setVibration(0);
    vibrationStopMs = 0;
  }
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

void normalize(Quaternion& value) {
  const float magnitude = sqrtf(value.w * value.w + value.x * value.x +
                                value.y * value.y + value.z * value.z);
  if (magnitude <= 0.000001f) {
    value = {};
    return;
  }
  value.w /= magnitude;
  value.x /= magnitude;
  value.y /= magnitude;
  value.z /= magnitude;
}

Quaternion quaternionMultiply(const Quaternion& a, const Quaternion& b) {
  return { a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
           a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
           a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
           a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w };
}

void applyScreenTouchRotation(float angleX, float angleY, float angleZ) {
  const float angle = sqrtf(angleX * angleX + angleY * angleY +
                            angleZ * angleZ);
  if (angle <= 0.000001f) return;
  const float halfAngle = angle * 0.5f;
  const float scale = sinf(halfAngle) / angle;
  const Quaternion delta = { cosf(halfAngle), angleX * scale,
                             angleY * scale, angleZ * scale };
  // Left multiplication makes every incremental rotation screen-relative.
  touchOrientation = quaternionMultiply(delta, touchOrientation);
  normalize(touchOrientation);
}

void updateViewRotationCache() {
  const float sinAuto = sinf(autoRotationY);
  const float cosAuto = cosf(autoRotationY);
  const float sinLevel = sinf(levelRotationZ);
  const float cosLevel = cosf(levelRotationZ);

  // Base pose: gradual gravity leveling after automatic spin.
  const float b00 = cosLevel * cosAuto;
  const float b01 = -sinLevel;
  const float b02 = -cosLevel * sinAuto;
  const float b10 = sinLevel * cosAuto;
  const float b11 = cosLevel;
  const float b12 = -sinLevel * sinAuto;
  const float b20 = sinAuto;
  const float b21 = 0;
  const float b22 = cosAuto;

  // Touch pose is independent and is always applied in screen space.
  const float xx = touchOrientation.x * touchOrientation.x;
  const float yy = touchOrientation.y * touchOrientation.y;
  const float zz = touchOrientation.z * touchOrientation.z;
  const float xy = touchOrientation.x * touchOrientation.y;
  const float xz = touchOrientation.x * touchOrientation.z;
  const float yz = touchOrientation.y * touchOrientation.z;
  const float wx = touchOrientation.w * touchOrientation.x;
  const float wy = touchOrientation.w * touchOrientation.y;
  const float wz = touchOrientation.w * touchOrientation.z;
  const float t00 = 1.0f - 2.0f * (yy + zz);
  const float t01 = 2.0f * (xy - wz);
  const float t02 = 2.0f * (xz + wy);
  const float t10 = 2.0f * (xy + wz);
  const float t11 = 1.0f - 2.0f * (xx + zz);
  const float t12 = 2.0f * (yz - wx);
  const float t20 = 2.0f * (xz - wy);
  const float t21 = 2.0f * (yz + wx);
  const float t22 = 1.0f - 2.0f * (xx + yy);

  viewM00 = t00 * b00 + t01 * b10 + t02 * b20;
  viewM01 = t00 * b01 + t01 * b11 + t02 * b21;
  viewM02 = t00 * b02 + t01 * b12 + t02 * b22;
  viewM10 = t10 * b00 + t11 * b10 + t12 * b20;
  viewM11 = t10 * b01 + t11 * b11 + t12 * b21;
  viewM12 = t10 * b02 + t11 * b12 + t12 * b22;
  viewM20 = t20 * b00 + t21 * b10 + t22 * b20;
  viewM21 = t20 * b01 + t21 * b11 + t22 * b21;
  viewM22 = t20 * b02 + t21 * b12 + t22 * b22;
}

Vec3 rotateView(const Vec3& source) {
  return { source.x * viewM00 + source.y * viewM01 + source.z * viewM02,
           source.x * viewM10 + source.y * viewM11 + source.z * viewM12,
           source.x * viewM20 + source.y * viewM21 + source.z * viewM22 };
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
  vertices.reserve(kVertexCount);
  quads.reserve(kQuadCount);
  int vertexOffset = 0;
  for (int face = 0; face < 6; ++face) {
    const Vec3& bitangent = kFaceBitangents[face];
    for (int y = 0; y <= kGridSize; ++y) {
      for (int x = 0; x <= kGridSize; ++x) {
        const float u = x / static_cast<float>(kGridSize) * 2.0f - 1.0f;
        const float v = y / static_cast<float>(kGridSize) * 2.0f - 1.0f;
        Vec3 point = add(kFaceNormals[face],
                         add(multiply(kFaceTangents[face], u),
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
  visibleFrontNeon.reserve(quads.size() / 2);
}

int findClosestQuad(const Vec3& position) {
  const float absX = fabsf(position.x);
  const float absY = fabsf(position.y);
  const float absZ = fabsf(position.z);

  int face;
  if (absX >= absY && absX >= absZ) {
    face = position.x >= 0.0f ? 0 : 1;
  } else if (absY >= absZ) {
    face = position.y >= 0.0f ? 2 : 3;
  } else {
    face = position.z >= 0.0f ? 4 : 5;
  }

  const float normalDistance = dot(position, kFaceNormals[face]);
  if (normalDistance <= 0.0f) return -1;
  const float u = dot(position, kFaceTangents[face]) / normalDistance;
  const float v = dot(position, kFaceBitangents[face]) / normalDistance;
  const int x = std::max(0, std::min(
      kGridSize - 1,
      static_cast<int>((u + 1.0f) * 0.5f * kGridSize)));
  const int y = std::max(0, std::min(
      kGridSize - 1,
      static_cast<int>((v + 1.0f) * 0.5f * kGridSize)));
  return face * kGridSize * kGridSize + y * kGridSize + x;
}

void resetGame() {
  rngState = esp_random();
  if (!rngState) rngState = millis() | 1;
  autoRotationY = randomFloat(0.0f, kTau);
  touchOrientation = {};
  touchVelocityX = 0;
  touchVelocityY = 0;
  touchVelocityZ = 0;
  updateViewRotationCache();
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

float angleDifference(float target, float current) {
  return atan2f(sinf(target - current), cosf(target - current));
}

void updateImu() {
  const uint32_t now = millis();
  frameDeltaSeconds = (now - previousImuMs) * 0.001f;
  previousImuMs = now;
  frameDeltaSeconds = std::max(0.001f, std::min(0.05f, frameDeltaSeconds));
  if (!M5.Imu.isEnabled()) return;

  float accelX = 0;
  float accelY = 0;
  float accelZ = 0;
  if (!M5.Imu.getAccel(&accelX, &accelY, &accelZ)) return;
  if (!levelInitialized) {
    filteredAccelX = accelX;
    filteredAccelY = accelY;
    filteredAccelZ = accelZ;
    levelInitialized = true;
  } else {
    filteredAccelX += (accelX - filteredAccelX) * kGravityFilter;
    filteredAccelY += (accelY - filteredAccelY) * kGravityFilter;
    filteredAccelZ += (accelZ - filteredAccelZ) * kGravityFilter;
  }

  // StopWatch's IMU is rotated 90 degrees from the display. These are the
  // screen-space components used by Avatar Mic as well.
  const float gravityScreenX = -filteredAccelY;
  const float gravityScreenY = -filteredAccelX;
  const float inPlaneGravity = sqrtf(
      gravityScreenX * gravityScreenX + gravityScreenY * gravityScreenY);
  float uprightBlend = (inPlaneGravity - kUprightBeginG) /
                       (kUprightFullG - kUprightBeginG);
  uprightBlend = std::max(0.0f, std::min(1.0f, uprightBlend));
  uprightBlend = uprightBlend * uprightBlend * (3.0f - 2.0f * uprightBlend);

  // When upright, point the south pole along gravity. When lying flat,
  // uprightBlend becomes zero and the south pole returns to screen bottom.
  const float gravityAngle = atan2f(-gravityScreenX, gravityScreenY);
  const float targetLevelZ = angleDifference(gravityAngle, 0.0f) * uprightBlend;
  levelAngularVelocityZ +=
      angleDifference(targetLevelZ, levelRotationZ) *
      kLevelSpringPerSecond2 * frameDeltaSeconds;
  const float damping = std::max(
      0.0f, 1.0f - kLevelDampingPerSecond * frameDeltaSeconds);
  levelAngularVelocityZ *= damping;
  levelAngularVelocityZ = std::max(
      -kMaxLevelRateRadiansPerSecond,
      std::min(kMaxLevelRateRadiansPerSecond, levelAngularVelocityZ));
  levelRotationZ += levelAngularVelocityZ * frameDeltaSeconds;
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
  const bool wasTouching = touching;
  if (M5.Touch.getCount()) {
    const auto detail = M5.Touch.getDetail(0);
    if (detail.isPressed() || detail.wasPressed()) {
      if (touching) {
        const float dragRight = detail.x - previousX;
        const float dragDown = detail.y - previousY;
        const float deltaX = dragDown * kTouchRadiansPerPixel;
        const float deltaY = -dragRight * kTouchRadiansPerPixel;
        applyScreenTouchRotation(deltaX, deltaY, 0);
        touchVelocityX = deltaX / frameDeltaSeconds;
        touchVelocityY = deltaY / frameDeltaSeconds;
        touchVelocityZ = 0;
      } else {
        touchVelocityX = 0;
        touchVelocityY = 0;
        touchVelocityZ = 0;
      }
      previousX = detail.x;
      previousY = detail.y;
      touching = true;
    }
  } else {
    touching = false;
  }

  if (wasTouching && !touching) touchReleasedMs = millis();

  if (!touching) {
    const bool returning = millis() - touchReleasedMs >= kTouchReturnDelayMs;
    const float damping = returning ? kTouchReturnDampingPerSecond
                                    : kTouchCoastDampingPerSecond;
    if (returning) {
      Quaternion shortest = touchOrientation;
      if (shortest.w < 0) {
        shortest.w = -shortest.w;
        shortest.x = -shortest.x;
        shortest.y = -shortest.y;
        shortest.z = -shortest.z;
      }
      const float sinHalf = sqrtf(shortest.x * shortest.x +
                                  shortest.y * shortest.y +
                                  shortest.z * shortest.z);
      if (sinHalf > 0.000001f) {
        const float angle = 2.0f * atan2f(sinHalf, shortest.w);
        const float spring = kTouchReturnSpringPerSecond2 *
                             frameDeltaSeconds * angle / sinHalf;
        touchVelocityX -= shortest.x * spring;
        touchVelocityY -= shortest.y * spring;
        touchVelocityZ -= shortest.z * spring;
      }
    }
    const float velocityDecay = std::max(
        0.0f, 1.0f - damping * frameDeltaSeconds);
    touchVelocityX *= velocityDecay;
    touchVelocityY *= velocityDecay;
    touchVelocityZ *= velocityDecay;
    applyScreenTouchRotation(touchVelocityX * frameDeltaSeconds,
                             touchVelocityY * frameDeltaSeconds,
                             touchVelocityZ * frameDeltaSeconds);
  }

  autoRotationY += kAutoSpinRadiansPerSecond * frameDeltaSeconds;
  if (autoRotationY > kTau) autoRotationY -= kTau;
  updateViewRotationCache();
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
    Quad& quad = quads[i];
    quad.depth =
        (rotatedVertices[quad.vertex[0]].z +
         rotatedVertices[quad.vertex[1]].z +
         rotatedVertices[quad.vertex[2]].z +
         rotatedVertices[quad.vertex[3]].z) * 0.25f;
    quadOrder[i] = i;
  }
  std::sort(quadOrder.begin(), quadOrder.end(), [](uint16_t a, uint16_t b) {
    return quads[a].depth > quads[b].depth;
  });

  visibleFrontNeon.clear();
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
  canvas.pushSprite(canvasOffsetX, canvasOffsetY);
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
  M5.Power.setVibration(0);
  // Serial.begin(115200);  // Enable together with reportFps() for profiling.
  M5.Display.setBrightness(100);
  screenWidth = M5.Display.width();
  screenHeight = M5.Display.height();
  canvasWidth = std::min<int16_t>(kCanvasSize, screenWidth);
  canvasHeight = std::min<int16_t>(kCanvasSize, screenHeight);
  canvasOffsetX = (screenWidth - canvasWidth) / 2;
  canvasOffsetY = (screenHeight - canvasHeight) / 2;
  centerX = canvasWidth * 0.5f;
  centerY = canvasHeight * 0.5f;
  const float canvasScale = canvasWidth / kCanvasScaleReference;
  sphereRadius = std::min(screenWidth, screenHeight) * 0.485f * canvasScale;
  fov = sphereRadius * 2.0f;
  createQuadSphere();
  M5.Display.fillScreen(TFT_BLACK);
  canvas.setColorDepth(16);
  canvas.setPsram(false);
  if (!canvas.createSprite(canvasWidth, canvasHeight)) {
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setTextColor(TFT_RED);
    M5.Display.setTextDatum(middle_center);
    M5.Display.drawString("Canvas allocation failed", screenWidth * 0.5f,
                          screenHeight * 0.5f);
    while (true) M5.delay(1000);
  }
  resetGame();
  previousImuMs = millis();
}

void loop() {
  M5.update();
  updateHaptic();
  if (M5.BtnA.wasClicked()) {
    resetGame();
    triggerHaptic(180, 50);
  }
  if (M5.BtnB.wasClicked()) {
    colorIndex = (colorIndex + 1) % 6;
    triggerHaptic(180, 50);
  }
  updateImu();
  updateTouchRotation();
  updateAgents();
  drawScene();
  // reportFps();  // Enable together with Serial.begin() for profiling.
  M5.delay(1);
}
