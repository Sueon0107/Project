#ifndef INC_MPU6050_H_
#define INC_MPU6050_H_

#include "stm32f1xx_hal.h"

// MPU6050 I2C 주소 (AD0=GND → 0x68, HAL용 << 1)
#define MPU6050_ADDR        0xD0

// 레지스터
#define MPU6050_PWR_MGMT_1  0x6B
#define MPU6050_SMPLRT_DIV  0x19
#define MPU6050_CONFIG      0x1A
#define MPU6050_GYRO_CONFIG 0x1B
#define MPU6050_ACCEL_CONFIG 0x1C
#define MPU6050_ACCEL_XOUT_H 0x3B
#define MPU6050_GYRO_XOUT_H  0x43

typedef struct {
    float accel_x, accel_y, accel_z;  // 단위: g
    float gyro_x,  gyro_y,  gyro_z;   // 단위: deg/s
} MPU6050_Data;

HAL_StatusTypeDef MPU6050_Init(I2C_HandleTypeDef *hi2c);
HAL_StatusTypeDef MPU6050_Read(I2C_HandleTypeDef *hi2c, MPU6050_Data *data);

#endif
