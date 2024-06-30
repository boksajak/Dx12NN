#include "shared.h"

// =========================================================================
//   Resources
// =========================================================================

// Constant buffer with data needed for NN
cbuffer NNDataCB : register(b0)
{
    NNData gData;
}

// Output image
RWTexture2D<float4> Output : register(u0);

// Network weights and biases
RWStructuredBuffer<float> nnWeights : register(u1);
RWStructuredBuffer<float> nnBiases : register(u2);

// Gradients for training
RWStructuredBuffer<int> gradientWeights : register(u3);
RWStructuredBuffer<int> gradientBiases : register(u4);

// Adam optimizer data
RWStructuredBuffer<float> adamWeightsMeans : register(u5);
RWStructuredBuffer<float> adamWeightsVariances : register(u6);
RWStructuredBuffer<float> adamBiasesMeans : register(u7);
RWStructuredBuffer<float> adamBiasesVariances : register(u8);

// Reference texture (input image)
Texture2D<float4> targetTexture: register(t0);

// =========================================================================
//   Network configuration
// =========================================================================

static const uint neuronsPerLayer[MAX_LAYERS] = { 
    NEURONS_PER_LAYER_0, NEURONS_PER_LAYER_1, NEURONS_PER_LAYER_2, NEURONS_PER_LAYER_3, NEURONS_PER_LAYER_4
};

// Base offsets of connection data per layer
static const uint connectionDataBaseOffsets[MAX_LAYERS] = { 
    CONNECTION_OFFSET_0, CONNECTION_OFFSET_1, CONNECTION_OFFSET_2, CONNECTION_OFFSET_3, CONNECTION_OFFSET_4
};

// Base offsets of neuron data per layer
static const uint neuronDataBaseOffsets[MAX_LAYERS] = { 
    NEURON_OFFSET_0, NEURON_OFFSET_1, NEURON_OFFSET_2, NEURON_OFFSET_3, NEURON_OFFSET_4
};

// =========================================================================
//   RNG
// =========================================================================

// 32-bit Xorshift random number generator
uint xorshift(inout uint rngState)
{
    rngState ^= rngState << 13;
    rngState ^= rngState >> 17;
    rngState ^= rngState << 5;
    return rngState;
}

// Jenkins's "one at a time" hash function
uint jenkinsHash(uint x)
{
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;
    return x;
}

// Converts unsigned integer into float int range <0; 1) by using 23 most significant bits for mantissa
float uintToFloat(uint x)
{
    return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}

// Initialize RNG for given pixel, and frame number (Xorshift-based version)
uint initRNG(uint2 pixelCoords, uint2 resolution, uint frameNumber)
{
    uint seed = dot(pixelCoords, uint2(1, resolution.x)) ^ jenkinsHash(frameNumber);
    return jenkinsHash(seed);
}

// Return random float in <0; 1) range (Xorshift-based version)
float rand(inout uint rngState)
{
    return uintToFloat(xorshift(rngState));
}

float randInRange(inout uint rng, float range)
{
    return (rand(rng) * 2.0f - 1.0f) * range;
}

// =========================================================================
//   Activation functions
// =========================================================================

#ifndef ACTIVATION_FUNCTION
    #define ACTIVATION_FUNCTION leakyRelu
#endif

#ifndef ACTIVATION_FUNCTION_DERIV
    #define ACTIVATION_FUNCTION_DERIV leakyReluDeriv
#endif

#define LEAKY_RELU_SLOPE 0.01f

float relu(float x)
{
    return max(0.0f, x);
}

float reluDeriv(float x)
{
    return (x <= 0.0f) ? (0.0f) : (1.0f);
}

float leakyRelu(float x)
{
    return (x >= 0.0f) ? x : (x * LEAKY_RELU_SLOPE);
}

float leakyReluDeriv(float x)
{
    return (x <= 0.0f) ? (LEAKY_RELU_SLOPE) : (1.0f);
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float sigmoidDeriv(float x)
{
    return x * (1.0f - x);
}

// =========================================================================
//   Helper functions
// =========================================================================

#define FLOAT_PACKING_CONSTANT 1000000.0f
int packFloat(float x)
{
    return int(x * FLOAT_PACKING_CONSTANT);
}

float unpackFloat(int x)
{
    return float(x) / FLOAT_PACKING_CONSTANT;
}

uint getNeuronDataIndex(uint layer, uint neuron)
{
    return neuronDataBaseOffsets[layer] + neuron;
}

uint getConnectionDataIndex(uint layer, uint neuronFrom, uint neuronTo)
{
    return connectionDataBaseOffsets[layer] + (neuronTo * neuronsPerLayer[layer - 1]) + neuronFrom;
}

// =========================================================================
//   Input Encoding functions
// =========================================================================

// Source: "Positional Encoding" from "NeRF: Representing Scenes as Neural RadianceFields for View Synthesis"
// https://arxiv.org/pdf/2003.08934
void frequencyEncoding(const float2 input, inout float activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER])
{
    const int inputCount = 2;
    
    int index = 0;
    [unroll]
    for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
    {
        const float p = PI * input[inputIndex];
        int modifier = 1;
        
        [unroll]
        for (int f = 0; f < NUM_FREQUENCIES; f++)
        {
            const float x = modifier * p;
            activations[index++] = sin(x);
            activations[index++] = cos(x);
            modifier *= 2;
        }
    }
}

void encodeInput(const float2 input, inout float activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER])
{
#if USE_IDENTITY_ENCODING
    
    // Identity encoding passes input as it is
    activations[0] = input.x;
    activations[1] = input.y;
    
#else // if USE_FREQUENCY_ENCODING
    
    frequencyEncoding(input, activations);
    
#endif
}

// =========================================================================
//   Inference
// =========================================================================

void forwardPass(float2 input, inout float activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER])
{
    // Encode input into first layer activations
    encodeInput(input, activations);
    
    // Calculate activations for every layer, going forward through the MLP network
    [unroll]
    for (uint layer = 1; layer < LAYER_COUNT; layer++)
    {
        const uint neuronCountCurrentLayer = neuronsPerLayer[layer];
        const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];
   
        [unroll(MAX_NEURONS_PER_LAYER)]
        for (uint neuron = 0; neuron < neuronCountCurrentLayer; neuron++)
        {
            const uint neuronDataIndex = getNeuronDataIndex(layer, neuron);
            
            // Evaluate neuron activation
            float neuronValue = nnBiases[neuronDataIndex];
            
            // Accumulate weighted contribution from all neurons connected to this neuron in previous layer
            for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
            {
                const uint weightDataIndex = getConnectionDataIndex(layer, previousNeuron, neuron);
                const uint previousNeuronDataIndex = getNeuronDataIndex(layer - 1, previousNeuron);
                
                neuronValue += nnWeights[weightDataIndex] * activations[previousNeuronDataIndex];
            }
            
            activations[neuronDataIndex] = ACTIVATION_FUNCTION(neuronValue);
        }
    }
}

// Runs inference to produce image output using the current network state
[numthreads(8, 8, 1)]
void Inference(
	int2 groupID : SV_GroupID,
	int2 groupThreadID : SV_GroupThreadID,
	int2 LaunchIndex : SV_DispatchThreadID)
{
    if (LaunchIndex.x >= gData.outputWidth || LaunchIndex.y >= gData.outputHeight)
        return;
    
    // Figure out UV coordinates for this pixel
    const float2 uvs = float2(LaunchIndex) / float2(gData.outputWidth - 1, gData.outputHeight - 1);
    
    // Do forward pass through the neural network with this pixel's UV coordinates as input
    float activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER];
    forwardPass(uvs, activations);
    
    // Map NN output to RGB result
    const uint outputLayerActivationIndex = getNeuronDataIndex(LAYER_COUNT - 1, 0);
    const float3 result = float3(activations[outputLayerActivationIndex + 0], 
                                 activations[outputLayerActivationIndex + 1], 
                                 activations[outputLayerActivationIndex + 2]);
    
    // Store output
    Output[LaunchIndex] = float4(result, 0);
}

// =========================================================================
//   Network initialization
// =========================================================================

// Source: "Normalized Initialization" from "Understanding the difficulty of training deep feedforward neural networks"
// https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
float xavierUniformScale(inout uint rng, const uint nInputs, const uint nOutputs)
{
    return sqrt(6.0f / (nInputs + nOutputs));
}

// This kernel initializes network to initial state
[numthreads(16, 16, 1)]
void Initialize(
	int2 groupID : SV_GroupID,
	int2 groupThreadID : SV_GroupThreadID,
	int2 LaunchIndex : SV_DispatchThreadID)
{
    const uint layer = LaunchIndex.x + 1;
    const uint neuron = LaunchIndex.y;
    if (layer >= LAYER_COUNT)
        return;
    
    const uint neuronCountCurrentLayer = neuronsPerLayer[layer];
    const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];
    if (neuron >= neuronCountCurrentLayer)
        return;
    
    // Initialize bias value to zero
    const uint neuronDataIndex = getNeuronDataIndex(layer, neuron);
    nnBiases[neuronDataIndex] = 0.0f;

    // Initialize Adam optimizer biases data (mean & variance)
    adamBiasesMeans[neuronDataIndex] = 0.0f;
    adamBiasesVariances[neuronDataIndex] = 0.0f;
    
    uint rng = initRNG(LaunchIndex, uint2(LAYER_COUNT, neuronCountCurrentLayer), gData.frameNumber);
    const float xavierWeightScale = xavierUniformScale(rng, neuronCountPreviousLayer, neuronCountCurrentLayer);
    
    // Initialize weights leading to this neuron
    for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
    {
        const float initialWeight = randInRange(rng, xavierWeightScale);
        const uint weightIndex = getConnectionDataIndex(layer, previousNeuron, neuron);
        nnWeights[weightIndex] = initialWeight;
        
        // Initialize Adam optimizer weights data (mean & variance)
        adamWeightsMeans[weightIndex] = 0.0f;
        adamWeightsVariances[weightIndex] = 0.0f;
    }
}

// =========================================================================
//   Network Training
// =========================================================================

void backpropagation(float3 target, float activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER])
{
    float errors[LAYER_COUNT * MAX_NEURONS_PER_LAYER];

    // Output layer derivatives
    {
        const uint neuronCountCurrentLayer = neuronsPerLayer[LAYER_COUNT - 1];
        const uint neuronCountPreviousLayer = neuronsPerLayer[LAYER_COUNT - 2];
   
        for (uint neuron = 0; neuron < neuronCountCurrentLayer; neuron++)
        {
            const uint neuronDataIndex = getNeuronDataIndex(LAYER_COUNT - 1, neuron);
            const float neuronActivation = activations[neuronDataIndex];
            const float dCost_O = 2.0f * (neuronActivation - target[neuron]);
            const float dO_Z = ACTIVATION_FUNCTION_DERIV(neuronActivation);
            const float dCost_Z = dCost_O * dO_Z;
            errors[neuronDataIndex] = dCost_Z;
            InterlockedAdd(gradientBiases[NonUniformResourceIndex(neuronDataIndex)], packFloat(dCost_Z));
            
            // Update weights
            for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
            {
                const uint previousNeuronDataIndex = getNeuronDataIndex(LAYER_COUNT - 2, previousNeuron);
                const float dCost_weight = dCost_Z * activations[previousNeuronDataIndex];
                const uint weightIndex = getConnectionDataIndex(LAYER_COUNT - 1, previousNeuron, neuron);
                InterlockedAdd(gradientWeights[NonUniformResourceIndex(weightIndex)], packFloat(dCost_weight));
            }
        }
    }
    
    // Hidden layer derivatives
    {
        [unroll(LAYER_COUNT - 2)]
        for (uint layer = LAYER_COUNT - 2; layer > 0; layer--)
        {
            const uint neuronCountCurrentLayer = neuronsPerLayer[layer];
            const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];
            const uint neuronCountNextLayer = neuronsPerLayer[layer + 1];
   
            for (uint neuron = 0; neuron < neuronCountCurrentLayer; neuron++)
            {
                float dCost_O = 0.0f;
                for (uint nextNeuron = 0; nextNeuron < neuronCountNextLayer; nextNeuron++)
                {
                    const uint weightIndex = getConnectionDataIndex(layer + 1, neuron, nextNeuron);
                    const uint nextNeuronDataIndex = getNeuronDataIndex(layer + 1, nextNeuron);
                    dCost_O += (errors[nextNeuronDataIndex] * nnWeights[NonUniformResourceIndex(weightIndex)]);
                }
                
                const uint neuronDataIndex = getNeuronDataIndex(layer, neuron);
                const float neuronActivation = activations[neuronDataIndex];
                const float dO_Z = ACTIVATION_FUNCTION_DERIV(neuronActivation);
                const float dCost_Z = dCost_O * dO_Z;
                errors[neuronDataIndex] = dCost_Z;
                InterlockedAdd(gradientBiases[NonUniformResourceIndex(neuronDataIndex)], packFloat(dCost_Z));

                // Update weights
                for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
                {
                    const uint previousNeuronDataIndex = getNeuronDataIndex(layer - 1, previousNeuron);
                    const float dCost_weight = dCost_Z * activations[previousNeuronDataIndex];
                    const uint weightIndex = getConnectionDataIndex(layer, previousNeuron, neuron);
                    InterlockedAdd(gradientWeights[NonUniformResourceIndex(weightIndex)], packFloat(dCost_weight));
                }
            }
        }
    }
}

// Runs training of the network (calculates gradients based on training inputs and reference output)
[numthreads(8, 1, 1)]
void Training(
	int2 groupID : SV_GroupID,
	int2 groupThreadID : SV_GroupThreadID,
	int2 LaunchIndex : SV_DispatchThreadID)
{
    if (LaunchIndex.x >= gData.batchSize || LaunchIndex.y > 0) return;
    
    // Initialize random numbers generator
    uint rng = initRNG(LaunchIndex, uint2(1, 1), gData.frameNumber);
    
    // Generate a random input (UV coordinates in the image)
    const float2 uvs = float2(rand(rng), rand(rng));
    
    // Load target value to learn for this input from reference image
    const float3 target = targetTexture[uvs * float2(gData.outputWidth - 1, gData.outputHeight - 1)].rgb;
    
    // First run forward pass to evaluate network activations for given input
    float activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER];
    forwardPass(uvs, activations);
    
    // Run backpropagation on current network state
    backpropagation(target, activations);
}

// Source: "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"
// https://arxiv.org/pdf/1412.6980
float AdamOptimizer(const float gradient, RWStructuredBuffer<float> means, RWStructuredBuffer<float> variances, const uint dataIndex)
{
    // Load mean and variance
    float mean = means[dataIndex];
    float variance = variances[dataIndex];
    
    // Update mean and variance for this training step
    mean = lerp(gradient, mean, gData.adamBeta1);
    variance = lerp((gradient * gradient), variance, gData.adamBeta2);
    
    // Calculate weight adjustment
    const float correctedMean = mean / (1.0f - gData.adamBeta1T);
    const float correctedVariance = variance / (1.0f - gData.adamBeta2T);
    const float weightAdjustment = -gData.learningRate * (correctedMean / (sqrt(correctedVariance) + gData.adamEpsilon));
    
    // Store updated mean and variance
    means[dataIndex] = mean;
    variances[dataIndex] = variance;
    
    return weightAdjustment;
}

// Runs optimization of the network (adjusts weights and biases using the calculated gradient)
[numthreads(16, 16, 1)]
void Optimize(
	int2 groupID : SV_GroupID,
	int2 groupThreadID : SV_GroupThreadID,
	int2 LaunchIndex : SV_DispatchThreadID)
{
    const uint layer = LaunchIndex.x + 1;
    const uint neuron = LaunchIndex.y;
    if (layer >= LAYER_COUNT) return;
    
    const uint neuronCountCurrentLayer = neuronsPerLayer[layer];
    const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];
    if (neuron >= neuronCountCurrentLayer) return;
    
    // Update bias value
    const uint neuronDataIndex = getNeuronDataIndex(layer, neuron);
    const float gradBias = unpackFloat(gradientBiases[neuronDataIndex]) * gData.rcpBatchSize;
    
    #if USE_SGD_OPTIMIZER
        nnBiases[neuronDataIndex] -= gradBias * gData.learningRate;
    #else //if USE_ADAM_OPTIMIZER
        nnBiases[neuronDataIndex] += AdamOptimizer(gradBias, adamBiasesMeans, adamBiasesVariances, neuronDataIndex);
    #endif
    
    // Update weights leading to this neuron
    for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
    {
        const uint weightIndex = getConnectionDataIndex(layer, previousNeuron, neuron);
        const float gradWeight = unpackFloat(gradientWeights[weightIndex]) * gData.rcpBatchSize;
    
        #if USE_SGD_OPTIMIZER
            nnWeights[weightIndex] -= gradWeight * gData.learningRate;
        #else //if USE_ADAM_OPTIMIZER
            nnWeights[weightIndex] += AdamOptimizer(gradWeight, adamWeightsMeans, adamWeightsVariances, weightIndex);
        #endif
    }
}

// This kernel clears gradient buffers to zeroes
[numthreads(16, 16, 1)]
void ClearGradients(
	int2 groupID : SV_GroupID,
	int2 groupThreadID : SV_GroupThreadID,
	int2 LaunchIndex : SV_DispatchThreadID)
{
    const uint layer = LaunchIndex.x + 1;
    const uint neuron = LaunchIndex.y;
    if (layer >= LAYER_COUNT)
        return;
    
    const uint neuronCountCurrentLayer = neuronsPerLayer[layer];
    const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];
    if (neuron >= neuronCountCurrentLayer)
        return;
    
    // Clear bias value
    const uint neuronDataIndex = getNeuronDataIndex(layer, neuron);
    gradientBiases[neuronDataIndex] = 0.0f;

    // Clear weights leading to this neuron
    for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
    {
        const uint weightIndex = getConnectionDataIndex(layer, previousNeuron, neuron);
        gradientWeights[weightIndex] = 0.0f;
    }
}