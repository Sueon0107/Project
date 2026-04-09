#ifndef INC_ANGLE_H_
#define INC_ANGLE_H_

#include "mpu6050.h"

// 상보필터 계수 (0.98 = 자이로 신뢰, 0.02 = 가속도계 보정)
#define COMP_FILTER_ALPHA  0.98f

typedef struct {
    float x;  // X축 기울기 (deg)
    float y;  // Y축 기울기 (deg)
} Angle_Data;

void Angle_Init(MPU6050_Data *imu, Angle_Data *angle);
void Angle_Update(MPU6050_Data *imu, Angle_Data *angle, float dt);

#endif
