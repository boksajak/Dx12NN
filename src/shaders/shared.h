
#if __cplusplus
#pragma once

	// Typedefs for sharing types between C++ (using GLM) and HLSL
	#include "..\common.h"
	typedef uint32_t uint;

	typedef glm::vec2 float2;
	typedef glm::vec3 float3;
	typedef glm::vec4 float4;

	typedef glm::uvec2 uint2;
	typedef glm::uvec3 uint3;
	typedef glm::uvec4 uint4;

	typedef glm::ivec2 int2;
	typedef glm::ivec3 int3;
	typedef glm::ivec4 int4;

	typedef glm::mat2 float2x2;
	typedef glm::mat3 float3x3;
	typedef glm::mat4 float4x4;
#endif

#define MAX_NEURONS_PER_LAYER 64
#define MAX_LAYERS 5

struct NNData
{
	uint frameNumber;
	uint outputWidth;
	uint outputHeight;
	float learningRate;

	float rcpBatchSize;
	uint batchSize;
	float adamBeta1;
	float adamBeta2;

	float adamEpsilon;
	float adamBeta1T;
	float adamBeta2T;
	uint frequencies;
};