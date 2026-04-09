#include "mpu6050.h"

HAL_StatusTypeDef MPU6050_Init(I2C_HandleTypeDef *hi2c)
{
    uint8_t data;
    HAL_StatusTypeDef ret;

    // 슬립 해제
    data = 0x00;
    ret = HAL_I2C_Mem_Write(hi2c, MPU6050_ADDR, MPU6050_PWR_MGMT_1, 1, &data, 1, 100);
    if (ret != HAL_OK) return ret;

    // 샘플링 레이트: 1kHz / (1+4) = 200Hz
    data = 0x04;
    ret = HAL_I2C_Mem_Write(hi2c, MPU6050_ADDR, MPU6050_SMPLRT_DIV, 1, &data, 1, 100);
    if (ret != HAL_OK) return ret;

    // 저역통과 필터: 42Hz
    data = 0x03;
    ret = HAL_I2C_Mem_Write(hi2c, MPU6050_ADDR, MPU6050_CONFIG, 1, &data, 1, 100);
    if (ret != HAL_OK) return ret;

    // 자이로 범위: ±250 deg/s
    data = 0x00;
    ret = HAL_I2C_Mem_Write(hi2c, MPU6050_ADDR, MPU6050_GYRO_CONFIG, 1, &data, 1, 100);
    if (ret != HAL_OK) return ret;

    // 가속도 범위: ±2g
    data = 0x00;
    ret = HAL_I2C_Mem_Write(hi2c, MPU6050_ADDR, MPU6050_ACCEL_CONFIG, 1, &data, 1, 100);
    if (ret != HAL_OK) return ret;

    return HAL_OK;
}

HAL_StatusTypeDef MPU6050_Read(I2C_HandleTypeDef *hi2c, MPU6050_Data *data)
{
    uint8_t buf[14];
    HAL_StatusTypeDef ret;

    ret = HAL_I2C_Mem_Read(hi2c, MPU6050_ADDR, MPU6050_ACCEL_XOUT_H, 1, buf, 14, 100);
    if (ret != HAL_OK) return ret;

    int16_t raw_ax = (int16_t)(buf[0]  << 8 | buf[1]);
    int16_t raw_ay = (int16_t)(buf[2]  << 8 | buf[3]);
    int16_t raw_az = (int16_t)(buf[4]  << 8 | buf[5]);
    int16_t raw_gx = (int16_t)(buf[8]  << 8 | buf[9]);
    int16_t raw_gy = (int16_t)(buf[10] << 8 | buf[11]);
    int16_t raw_gz = (int16_t)(buf[12] << 8 | buf[13]);

    data->accel_x = raw_ax / 16384.0f;
    data->accel_y = raw_ay / 16384.0f;
    data->accel_z = raw_az / 16384.0f;
    data->gyro_x  = raw_gx / 131.0f;
    data->gyro_y  = raw_gy / 131.0f;
    data->gyro_z  = raw_gz / 131.0f;

    return HAL_OK;
}
