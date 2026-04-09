#include "angle.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// 처음 한 번 가속도계로 초기 각도 설정
void Angle_Init(MPU6050_Data *imu, Angle_Data *angle)
{
    angle->x = atan2f(-imu->accel_x, imu->accel_z) * 180.0f / M_PI;
    angle->y = atan2f(-imu->accel_y, imu->accel_z) * 180.0f / M_PI;
}

// dt: 루프 주기 (초 단위)
// angle->x: X축 기울기, angle->y: Y축 기울기 (도 단위, 0 = 수평)
void Angle_Update(MPU6050_Data *imu, Angle_Data *angle, float dt)
{
    float accel_x = atan2f(-imu->accel_x, imu->accel_z) * 180.0f / M_PI;
    float accel_y = atan2f(-imu->accel_y, imu->accel_z) * 180.0f / M_PI;

    angle->x = COMP_FILTER_ALPHA * (angle->x - imu->gyro_y * dt)
             + (1.0f - COMP_FILTER_ALPHA) * accel_x;
    angle->y = COMP_FILTER_ALPHA * (angle->y - imu->gyro_x * dt)
             + (1.0f - COMP_FILTER_ALPHA) * accel_y;
}
