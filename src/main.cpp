#include "common.h"
#include "utils.h"
#include "shaders/shared.h"

// Windows DPI Scaling
#include <ShellScalingApi.h>
#pragma comment(lib, "shcore.lib")

const unsigned int frameWidth = 1820;
const unsigned int frameHeight = 980;

static const D3D12_HEAP_PROPERTIES UploadHeapProperties =
{
	D3D12_HEAP_TYPE_UPLOAD,
	D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
	D3D12_MEMORY_POOL_UNKNOWN,
	0, 0
};

static const D3D12_HEAP_PROPERTIES DefaultHeapProperties =
{
	D3D12_HEAP_TYPE_DEFAULT,
	D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
	D3D12_MEMORY_POOL_UNKNOWN,
	0, 0
};

static const D3D12_HEAP_PROPERTIES ReadbackHeapProperties =
{
	D3D12_HEAP_TYPE_READBACK,
	D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
	D3D12_MEMORY_POOL_UNKNOWN,
	0, 0
};

struct D3D12ShaderInfo
{
	LPCWSTR		filename;
	LPCWSTR		entryPoint;
	LPCWSTR		targetProfile;
	LPCVOID	    data = nullptr;
	UINT32	    dataSize = 0;
	UINT32	    dataCodePage = 0;

	D3D12ShaderInfo(LPCWSTR inFilename, LPCWSTR inEntryPoint, LPCWSTR inProfile)
	{
		filename = inFilename;
		entryPoint = inEntryPoint;
		targetProfile = inProfile;
	}

	D3D12ShaderInfo(LPCWSTR inFilename, LPCVOID	inData, UINT32 inDataSize, UINT32 inDataCodePage, LPCWSTR inEntryPoint, LPCWSTR inProfile)
	{
		filename = inFilename;
		data = inData;
		dataSize = inDataSize;
		dataCodePage = inDataCodePage;
		entryPoint = inEntryPoint;
		targetProfile = inProfile;
	}

	D3D12ShaderInfo()
	{
		filename = NULL;
		entryPoint = NULL;
		targetProfile = NULL;
	}
};

struct D3D12Program
{
	D3D12Program(D3D12ShaderInfo shaderInfo)
	{
		info = shaderInfo;
		blob = nullptr;
		subobject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
		exportName = shaderInfo.entryPoint;
		exportDesc.ExportToRename = nullptr;
		exportDesc.Flags = D3D12_EXPORT_FLAG_NONE;
	}

	void SetBytecode()
	{
		if (exportName.empty())
		{
			// Export everything
			dxilLibDesc.NumExports = 0;
			dxilLibDesc.pExports = nullptr;
		}
		else
		{
			exportDesc.Name = exportName.c_str();
			dxilLibDesc.NumExports = 1;
			dxilLibDesc.pExports = &exportDesc;
		}
		dxilLibDesc.DXILLibrary.BytecodeLength = blob->GetBufferSize();
		dxilLibDesc.DXILLibrary.pShaderBytecode = blob->GetBufferPointer();

		subobject.pDesc = &dxilLibDesc;
	}

	D3D12Program()
	{
		blob = nullptr;
		exportDesc.ExportToRename = nullptr;
	}

	D3D12ShaderInfo		info = {};
	IDxcBlob* blob;

	D3D12_DXIL_LIBRARY_DESC	dxilLibDesc;
	D3D12_EXPORT_DESC		exportDesc;
	D3D12_STATE_SUBOBJECT	subobject;
	std::wstring			exportName;
};

struct D3D12ShaderCompiler
{
	dxc::DxcDllSupport		DxcDllHelper = {};
	IDxcCompiler* compiler = {};
	IDxcLibrary* library = {};

	D3D12ShaderCompiler()
	{
		compiler = nullptr;
		library = nullptr;
	}

	void Init()
	{
		HRESULT hr = DxcDllHelper.Initialize();
		utils::validate(hr, L"Failed to initialize DxCDllSupport!");

		hr = DxcDllHelper.CreateInstance(CLSID_DxcCompiler, &compiler);
		utils::validate(hr, L"Failed to create DxcCompiler!");

		hr = DxcDllHelper.CreateInstance(CLSID_DxcLibrary, &library);
		utils::validate(hr, L"Failed to create DxcLibrary!");
	}

	void CompileShader(D3D12Program& program, std::vector<std::wstring> compilerFlags = {})
	{
		CompileShader(program.info, &program.blob, compilerFlags);
		program.SetBytecode();
	}

	void CompileShader(D3D12ShaderInfo& info, IDxcBlob** blob, std::vector<std::wstring> compilerFlags = {})
	{
		HRESULT hr;
		IDxcOperationResult* result = nullptr;
		bool retryCompile = true;

		while (retryCompile) {

			UINT32 codePage(0);
			IDxcBlobEncoding* pShaderText(nullptr);

			// Load and encode the shader file
			if (info.dataSize == 0 || info.data == nullptr)
				hr = library->CreateBlobFromFile(info.filename, &codePage, &pShaderText);
			else
				hr = library->CreateBlobWithEncodingFromPinned(info.data, info.dataSize, info.dataCodePage, &pShaderText);

			utils::validate(hr, L"Error: failed to create blob from shader file!");

			// Create the compiler include handler
			IDxcIncludeHandler* dxcIncludeHandler;
			hr = library->CreateIncludeHandler(&dxcIncludeHandler);
			utils::validate(hr, L"Error: failed to create include handler");

			// Additional compiler flags (always present)
			compilerFlags.push_back(L"/all_resources_bound");
			compilerFlags.push_back(L"/enable-16bit-types");

			// Process compiler flags to an array of char* pointers
			LPCWSTR* arguments = new LPCWSTR[compilerFlags.size()];
			for (size_t i = 0; i < compilerFlags.size(); i++) arguments[i] = compilerFlags[i].c_str();

			// Compile the shader
			hr = compiler->Compile(pShaderText, info.filename, info.entryPoint, info.targetProfile, arguments, (UINT32)compilerFlags.size(), nullptr, 0, dxcIncludeHandler, &result);
			delete[] arguments;
			utils::validate(hr, L"Error: failed to compile shader!");

			// Verify the result 
			result->GetStatus(&hr);
			if (FAILED(hr))
			{
				IDxcBlobEncoding* error;
				hr = result->GetErrorBuffer(&error);
				utils::validate(hr, L"Error: failed to get shader compiler error buffer!");

				// Convert error blob to a string
				std::vector<char> infoLog(error->GetBufferSize() + 1);
				memcpy(infoLog.data(), error->GetBufferPointer(), error->GetBufferSize());
				infoLog[error->GetBufferSize()] = 0;

				std::string errorMsg = "Shader Compiler Error:\n";
				errorMsg.append(infoLog.data());

				if (MessageBoxA(nullptr, errorMsg.c_str(), "Error!", MB_RETRYCANCEL) == IDRETRY) {
					// Another retry
					continue;
				}
				else {
					// User canceled
					return;
				}
			}

			// Successful compilation
			retryCompile = false;
		}

		hr = result->GetResult(blob);
		utils::validate(hr, L"Error: failed to get shader blob result!");
	}


	void Destroy()
	{
		SAFE_RELEASE(compiler);
		SAFE_RELEASE(library);
		DxcDllHelper.Cleanup();
	}

};

class Profiler
{
public:

	void Initialize(ID3D12Device5* device) {

		// Create query heap
		{
			SAFE_RELEASE(mQueryHeap);

			D3D12_QUERY_HEAP_DESC heapDesc = { };
			heapDesc.Count = cMaxQueryCount * 2;
			heapDesc.NodeMask = 0;
			heapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
			HRESULT hr = device->CreateQueryHeap(&heapDesc, IID_PPV_ARGS(&mQueryHeap));
			utils::validate(hr, L"Error: failed to create profiling query heap!");
		}

		// Create readback heap for query results
		{
			SAFE_RELEASE(mReadbackBuffer);

			D3D12_RESOURCE_DESC resourceDesc = {};
			resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			resourceDesc.Width = uint32_t(cMaxQueryCount * 2 * sizeof(uint64_t));
			resourceDesc.Height = 1;
			resourceDesc.DepthOrArraySize = 1;
			resourceDesc.MipLevels = 1;
			resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
			resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
			resourceDesc.SampleDesc.Count = 1;
			resourceDesc.SampleDesc.Quality = 0;
			resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			resourceDesc.Alignment = 0;

			HRESULT	hr = device->CreateCommittedResource(&ReadbackHeapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc,
				D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&mReadbackBuffer));
			utils::validate(hr, L"Error: failed to create profiling readback heap!");
		}
	}

	std::string BeginFrame(ID3D12GraphicsCommandList4* cmdList, ID3D12CommandQueue* cmdQueue) {

		std::string result = "";

		// Get resolved timings from previous frame
		if (mQueryCount > 0) {

			// Download queries results
			D3D12_RANGE range;
			range.Begin = SIZE_T(0);
			range.End = SIZE_T(mQueryCount * 2 * sizeof(uint64_t));
			void* mappedData = nullptr;
			mReadbackBuffer->Map(0, &range, &mappedData);

			uint64_t* queryData = ((uint64_t*)mappedData);

			for (size_t i = 0; i < mQueryCount; i++)
			{
				uint64_t startTime = queryData[i * 2];
				uint64_t endTime = queryData[i * 2 + 1];

				uint64_t delta = endTime - startTime;
				double frequency = double(mLastGpuFrequency);
				float queryTimeMs = (delta / frequency) * 1000.0;

				char temp[256];
				snprintf(temp, 256, "%-15ls: %12.4fms\n", mQueryNames[i].c_str(), queryTimeMs);
				result += temp;
			}

			mReadbackBuffer->Unmap(0, nullptr);
		}

		mCmdList = cmdList;
		mQueryCount = 0;

		cmdQueue->GetTimestampFrequency(&mLastGpuFrequency);

		return result;
	}

	unsigned int StartEvent(std::wstring name) {

		unsigned int queryIndex = mQueryCount++;
		mQueryNames[queryIndex] = name;

		if (queryIndex == cMaxQueryCount) {
			utils::validate(E_FAIL, L"Only 'cMaxQueryCount' profiles are supported, this is one too many!");
			return -1;
		}

		// Timestamp query only supports EndQuery method (so we call it during StartEvent as well as EndEvent)
		mCmdList->EndQuery(mQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, queryIndex * 2);

		return queryIndex;
	}

	void StopEvent(unsigned int queryIndex)
	{
		mCmdList->EndQuery(mQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, queryIndex * 2 + 1);
		mCmdList->ResolveQueryData(mQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, queryIndex * 2, 2, mReadbackBuffer, queryIndex * 2 * sizeof(uint64_t));
	}

private:
	ID3D12GraphicsCommandList4* mCmdList = nullptr;
	ID3D12QueryHeap* mQueryHeap = nullptr;
	ID3D12Resource* mReadbackBuffer = nullptr;

	unsigned int mQueryCount = 0;
	static const unsigned int cMaxQueryCount = 32;
	std::wstring mQueryNames[cMaxQueryCount];
	uint64_t mLastGpuFrequency = 0;
};

class ProfileEvent
{
public:

	ProfileEvent(Profiler* profiler, std::wstring name) {
		mId = profiler->StartEvent(name);
		mProfiler = profiler;
	}

	~ProfileEvent() {
		mProfiler->StopEvent(mId);
	}
private:

	unsigned int mId;
	Profiler* mProfiler;
};

#define TOKENPASTE(x, y) x ## y
#define TOKENPASTE2(x, y) TOKENPASTE(x, y)
#define PROFILE(_name) ProfileEvent TOKENPASTE2(profilingContext, __COUNTER__)(&mProfiler, _name);

class Dx12NN 
{
public:
	void Initialize(HWND hwnd) 
	{
		mDpiScale = utils::getDpiScale(hwnd);
		mNNData.frameNumber = 0;
		mReloadShaders = false;
		mNNNeedsInitialization = true;

		mShaderCompiler.Init();

		initializeDx12(hwnd);
		initImGui(hwnd);

		mProfiler.Initialize(mDevice);
	}

	void ReloadShaders() {
		mReloadShaders = true;
	}

	bool Update(HWND hwnd)
	{
		std::string profiling = mProfiler.BeginFrame(mCmdList, mCmdQueue);

		// Start the Dear ImGui frame
		ImGui_ImplDX12_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		// Position the ImGui window on start to the right
		if (mNNData.frameNumber == 0) {
			ImGui::SetNextWindowPos(ImVec2(frameWidth - 450 - 20, 20));
			ImGui::SetNextWindowSize(ImVec2(450, frameHeight - 40.f));
		}

		ImGui::Begin("DX12 Neural Network");

		ImGui::Text("Adapter: %ls", mAdapterName.c_str());
		ImGui::Checkbox("VSync", &mEnableVSync);

		if (ImGui::Button("Load Image")) mRequestLoadFile = true;
		if (ImGui::Button("[F5]Reload Shaders")) mReloadShaders = true;
		if (ImGui::Button("Initialize Weights")) mNNNeedsInitialization = true;

		ImGui::Text("Training Steps: %i", mTrainingSteps);
		ImGui::Checkbox("Enable Learning", &mEnableTraining);
		if (!mEnableTraining) {
			if (ImGui::Button("Step")) mTrainingStep = true;
		}
		ImGui::SliderInt("Batch Size", &mBatchSize, 1, 4096);
		ImGui::SliderFloat("Learning Rate", &mLearningRate, 0.0f, mOptimizerType == OptimizerType::Adam ? 0.01f : 0.1f);
		
		if (ImGui::Combo("Optimizer", (int*)&mOptimizerType, "SGD\0Adam\0\0")) {
			mReloadShaders = true;
		}

		if (ImGui::Combo("Input Encoding", (int*)&mInputEncodingType, "Identity\0Frequency\0\0")) {
			mReloadShaders = true;
			mNNNeedsInitialization = true;
		}

		if (mInputEncodingType == InputEncodingType::Frequency) {
			if (ImGui::SliderInt("Frequencies", &mFrequencies, 2, 16)) {
				mNNArchitectureDirty = true;
			}
		}

		if (ImGui::Combo("Activation Function", (int*)&mActivationFunctionType, "ReLU\0Leaky ReLU\0Sigmoid\0\0")) {
			mReloadShaders = true;
			mNNNeedsInitialization = true;
		}

		if (ImGui::SliderInt("Layer Count", &mNewLayerCount, 3, MAX_LAYERS)) {
			mNNArchitectureDirty = true;
		}

		if (ImGui::SliderInt("Neurons Per Layer", &mNewNeuronsPerLayer, 1, MAX_NEURONS_PER_LAYER)) {
			mNNArchitectureDirty = true;
		}

		if (mNNArchitectureDirty) {
			ImGui::Text("Press Apply for changes to take effect");
			if (ImGui::Button("Apply"))
			{
				mReloadShaders = true;
				mNNNeedsInitialization = true;
				mNNArchitectureDirty = false;
				mLayerCount = mNewLayerCount;
				mNeuronsPerLayer = mNewNeuronsPerLayer;
			}
		}

		ImGui::TextUnformatted(profiling.c_str());

		if (mReloadShaders)
		{
			ImGui::SetWindowFontScale(2.0f);
			ImGui::Text("RECOMPILING SHADERS!");
			ImGui::SetWindowFontScale(1.0f);
		}

		ImGui::End();

		if (mTargetImageLoaded)
		{

			if (mNNNeedsInitialization)
			{
				mTrainingSteps = 0;
			}

			// Update constant buffer
			{
				mNNData.outputWidth = mTargetWidth;
				mNNData.outputHeight = mTargetHeight;
				mNNData.learningRate = mLearningRate;

				mNNData.batchSize = mBatchSize;
				mNNData.rcpBatchSize = 1.0f / float(mBatchSize);

				// Adam parameters
				mNNData.adamBeta1 = 0.9f;
				mNNData.adamBeta2 = 0.999f;
				mNNData.adamEpsilon = 0.00000001f;
				mNNData.adamBeta1T = glm::pow(mNNData.adamBeta1, mTrainingSteps + 1);
				mNNData.adamBeta2T = glm::pow(mNNData.adamBeta2, mTrainingSteps + 1);

				mNNData.frequencies = mFrequencies;
				uploadConstantBuffer();

				mNNData.frameNumber++;
			}

			// Setup root signature
			{
				ID3D12DescriptorHeap* ppHeaps[] = { mDescriptorHeap };
				mCmdList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
				mCmdList->SetComputeRootSignature(mGlobalRootSignature);
				mCmdList->SetComputeRootDescriptorTable(UINT(RootParameterIndex::CbvSrvUavs), mDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
			}

			// Run neural network
			{
				PROFILE(L"NN Total");

				// Initialize network weights
				if (mNNNeedsInitialization)
				{
					mNNNeedsInitialization = false;

					PROFILE(L"Initialize");

					const uint32_t dispatchWidth = utils::divRoundUp(mLayerCount, 16);
					const uint32_t dispatchHeight = utils::divRoundUp(MAX_NEURONS_PER_LAYER, 16);

					mCmdList->SetPipelineState(mInitializationPSO);
					mCmdList->Dispatch(dispatchWidth, dispatchHeight, 1);

					uavBarrier(mNNWeightsBuffer);
					uavBarrier(mNNBiasesBuffer);
				}

				// Run training
				if (mEnableTraining || mTrainingStep)
				{
					// Clear gradients
					{
						const uint32_t dispatchWidth = utils::divRoundUp(mLayerCount, 16);
						const uint32_t dispatchHeight = utils::divRoundUp(MAX_NEURONS_PER_LAYER, 16);

						mCmdList->SetPipelineState(mClearGradientsPSO);
						mCmdList->Dispatch(dispatchWidth, dispatchHeight, 1);
					}

					// Calculate gradients with backpropagation
					{
						PROFILE(L"Backpropagation");

						uavBarrier(mNNGradientWeightsBuffer);
						uavBarrier(mNNGradientBiasesBuffer);

						const uint32_t dispatchWidth = utils::divRoundUp(mBatchSize, 8);

						mCmdList->SetPipelineState(mTrainingPSO);
						mCmdList->Dispatch(dispatchWidth, 1, 1);
					}

					// Optimize network (apply gradients)
					{
						PROFILE(L"Optimization");

						uavBarrier(mNNGradientWeightsBuffer);
						uavBarrier(mNNGradientBiasesBuffer);

						const uint32_t dispatchWidth = utils::divRoundUp(mLayerCount, 16);
						const uint32_t dispatchHeight = utils::divRoundUp(MAX_NEURONS_PER_LAYER, 16);

						mCmdList->SetPipelineState(mOptimizePSO);
						mCmdList->Dispatch(dispatchWidth, dispatchHeight, 1);
					}

					mTrainingSteps++;
					mTrainingStep = false;
				}

				// Run inference
				{
					PROFILE(L"Inference");

					uavBarrier(mNNWeightsBuffer);
					uavBarrier(mNNBiasesBuffer);

					const uint32_t dispatchWidth = utils::divRoundUp(mTargetWidth, 8);
					const uint32_t dispatchHeight = utils::divRoundUp(mTargetHeight, 8);

					mCmdList->SetPipelineState(mInferencePSO);
					mCmdList->Dispatch(dispatchWidth, dispatchHeight, 1);
				}
			}

			// Clear the screen
			D3D12_CPU_DESCRIPTOR_HANDLE destination = getCurrentBackBufferView();
			const glm::vec4 black = glm::vec4(0, 0, 0, 0);
			transitionBarrier(mBackBuffer[mCurrentFrameIndex], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
			mCmdList->ClearRenderTargetView(destination, &black.x, 0, nullptr);

			// Specify the buffers we are going to render to - destination (back buffer)
			D3D12_CPU_DESCRIPTOR_HANDLE depthStencilBufferViewHandle = mDsvHeap->GetCPUDescriptorHandleForHeapStart();
			mCmdList->OMSetRenderTargets(1, &destination, true, &depthStencilBufferViewHandle);

			// Copy the final output and target texture to the back buffer
			{
				transitionBarrier(mBackBuffer[mCurrentFrameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
				transitionBarrier(mOutputBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
				transitionBarrier(mTargetTextureBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);

				// Copy Final output (inference result)
				{
					CD3DX12_TEXTURE_COPY_LOCATION dest(mBackBuffer[mCurrentFrameIndex]);
					CD3DX12_TEXTURE_COPY_LOCATION src(mOutputBuffer);
					CD3DX12_BOX box(0, 0, mTargetWidth, mTargetHeight);
					mCmdList->CopyTextureRegion(&dest, 512 + 250, 100, 0, &src, &box);
				}

				// Copy Target texture (training image)
				{
					CD3DX12_TEXTURE_COPY_LOCATION dest(mBackBuffer[mCurrentFrameIndex]);
					CD3DX12_TEXTURE_COPY_LOCATION src(mTargetTextureBuffer);
					CD3DX12_BOX box(0, 0, mTargetWidth, mTargetHeight);
					mCmdList->CopyTextureRegion(&dest, 100, 100, 0, &src, &box);
				}
				
				transitionBarrier(mBackBuffer[mCurrentFrameIndex], D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
				transitionBarrier(mOutputBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
				transitionBarrier(mTargetTextureBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
			}
		}

		// Render ImGUI
		{
			mCmdList->SetDescriptorHeaps(1, &imguiSrvDescHeap);
			ImGui::Render();
			ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), mCmdList);
		}

		transitionBarrier(mBackBuffer[mCurrentFrameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);

		// End frame - process cmd. list and move to next frame
		submitCmdList();
		waitForGPU();
		present();
		moveToNextFrame();
		resetCommandList();

		// Reload shaders here if needed
		if (mReloadShaders) {
			createComputePasses();
			mReloadShaders = false;
		}

		// Load image from file if needed
		if (mRequestLoadFile) {

			OPENFILENAME ofn;       // common dialog box structure
			wchar_t szFile[512];       // buffer for file name

			// Initialize OPENFILENAME
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hwnd;
			ofn.lpstrFile = szFile;
			// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
			// use the contents of szFile to initialize itself.
			ofn.lpstrFile[0] = '\0';
			ofn.nMaxFile = sizeof(szFile);
			ofn.lpstrFilter = L"All\0*.*\0";
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			std::wstring assetsFolder = utils::getExePath() + L"assets\\";
			ofn.lpstrInitialDir = assetsFolder.c_str();
			ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

			// Display the Open dialog box. 

			if (GetOpenFileName(&ofn) == TRUE)
				loadTargetImage(szFile);

			mRequestLoadFile = false;
			mNNNeedsInitialization = true;
		}

		return true; 
	}

	void Cleanup() 
	{
		ImGui_ImplWin32_Shutdown();
		ImGui::DestroyContext();
	}

private:

	void initializeDx12(HWND hwnd) {

		for (int i = 0; i < kMaxFramesInFlight; i++) {
			mFenceValues[i] = 0;
			mBackBuffer[i] = nullptr;
			mCmdAlloc[i] = nullptr;
		}

		// Create DXGI factory
		HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&mDxgiFactory));
		utils::validate(hr, L"Error: failed to create DXGI factory!");

		const D3D_FEATURE_LEVEL kDx12FeatureLevel = D3D_FEATURE_LEVEL_12_1;

#ifdef _DEBUG
		// Enable the D3D12 debug layer.
		{
			if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&mDebugController))))
			{
				mDebugController->EnableDebugLayer();
			}
		}
#endif

		// Find a suitable adapter to run D3D
		int i = 0;
		IDXGIAdapter1* adapter = nullptr;
		DXGI_ADAPTER_DESC1 selectedAdapterDesc = {};
		while (mDxgiFactory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND)
		{
			ID3D12Device5* tempDevice;

			if (SUCCEEDED(D3D12CreateDevice(adapter, kDx12FeatureLevel, _uuidof(ID3D12Device5), (void**)&tempDevice)))
			{
				// Check if the adapter supports ray tracing, but don't create the actual device yet.
				D3D12_FEATURE_DATA_D3D12_OPTIONS5 features = {};
				HRESULT hr = tempDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &features, sizeof(features));
				if (SUCCEEDED(hr))
				{
					DXGI_ADAPTER_DESC1 desc;
					adapter->GetDesc1(&desc);

					if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
						selectedAdapterDesc = desc;
						break;
					}
				}
				SAFE_RELEASE(tempDevice);
				adapter->Release();
				i++;
			}
		}

		// Create D3D Device
		{
			HRESULT hr = mDxgiFactory->EnumAdapterByLuid(selectedAdapterDesc.AdapterLuid, IID_PPV_ARGS(&mAdapter));
			utils::validate(hr, L"Error: failed to enumerate selected adapter by luid!");

			hr = D3D12CreateDevice(mAdapter, kDx12FeatureLevel, _uuidof(ID3D12Device5), (void**)&mDevice);
			utils::validate(hr, L"Error: failed to create D3D device!");

			mAdapterName = selectedAdapterDesc.Description;
		}

		// Create command queue
		{
			D3D12_COMMAND_QUEUE_DESC desc = {};
			desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
			desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

			HRESULT hr = mDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(&mCmdQueue));
			utils::validate(hr, L"Error: failed to create command queue!");
		}

		// Create a command allocator for each frame
		{
			for (UINT n = 0; n < kMaxFramesInFlight; n++)
			{
				HRESULT hr = mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&mCmdAlloc[n]));
				utils::validate(hr, L"Error: failed to create the command allocator!");
			}
		}

		// Create Command List
		{
			// Create the command list
			HRESULT hr = mDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, mCmdAlloc[mCurrentFrameIndex], nullptr, IID_PPV_ARGS(&mCmdList));
			hr = mCmdList->Close();
			utils::validate(hr, L"Error: failed to create the command list!");

			resetCommandList();
		}

		// Create fence
		{
			HRESULT hr = mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mFence));
			utils::validate(hr, L"Error: failed to create fence!");

			mFenceValues[mCurrentFrameIndex]++;

			// Create an event handle to use for frame synchronization
			mFenceEvent = CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS);
			if (mFenceEvent == nullptr)
			{
				hr = HRESULT_FROM_WIN32(GetLastError());
				utils::validate(hr, L"Error: failed to create fence event!");
			}
		}

		// Create swap chain
		{
			// Check for tearing support
			BOOL allowTearing = FALSE;
			HRESULT hr = mDxgiFactory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing));
			utils::validate(hr, L"Error: failed to create DXGI factory!");

			mIsTearingSupport = SUCCEEDED(hr) && allowTearing;

			// Describe the swap chain
			DXGI_SWAP_CHAIN_DESC1 desc = {};
			desc.BufferCount = kMaxFramesInFlight;
			desc.Width = frameWidth;
			desc.Height = frameHeight;
			desc.Format = mBackBufferFormat;
			desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
			desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
			desc.SampleDesc.Count = 1;
			desc.SampleDesc.Quality = 0;
			desc.Flags = mIsTearingSupport ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

			IDXGISwapChain1* tempSwapChain;

			// Create the swap chain
			hr = mDxgiFactory->CreateSwapChainForHwnd(mCmdQueue, hwnd, &desc, 0, 0, &tempSwapChain);
			utils::validate(hr, L"Error: failed to create swap chain!");

			if (mIsTearingSupport)
			{
				// When tearing support is enabled we will handle ALT+Enter key presses in the
				// window message loop rather than let DXGI handle it by calling SetFullscreenState.
				mDxgiFactory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
				utils::validate(hr, L"Error: failed to make window association!");
			}

			// Get the swap chain interface
			hr = tempSwapChain->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void**>(&mSwapChain));
			utils::validate(hr, L"Error: failed to cast swap chain!");

			SAFE_RELEASE(tempSwapChain);
			mCurrentFrameIndex = mSwapChain->GetCurrentBackBufferIndex();
		}

		// Create RTV heap
		{
			// Describe the RTV heap
			D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
			rtvDesc.NumDescriptors = kMaxFramesInFlight;
			rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
			rtvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

			// Create the RTV heap
			HRESULT hr = mDevice->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&mRtvHeap));
			utils::validate(hr, L"Error: failed to create RTV descriptor heap!");

			mRtvDescSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		}

		// Create DSV heap
		{
			D3D12_DESCRIPTOR_HEAP_DESC heapDescription;
			heapDescription.NumDescriptors = 1;
			heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
			heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
			heapDescription.NodeMask = 0;

			HRESULT result = mDevice->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(&mDsvHeap));
			utils::validate(result, L"Error: failed to create DSV heap!");
		}

		// Create back buffer
		{
			HRESULT hr;
			D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = mRtvHeap->GetCPUDescriptorHandleForHeapStart();

			// Create a RTV for each back buffer
			for (unsigned int n = 0; n < kMaxFramesInFlight; n++)
			{
				hr = mSwapChain->GetBuffer(n, IID_PPV_ARGS(&mBackBuffer[n]));
				utils::validate(hr, L"Error: failed to get swap chain buffer!");

				mDevice->CreateRenderTargetView(mBackBuffer[n], nullptr, rtvHandle);

				rtvHandle.ptr += mRtvDescSize;
			}
		}

		// Create depth stencil buffer
		{
			// Create the depth/stencil buffer and view.
			D3D12_RESOURCE_DESC depthStencilDesc;
			depthStencilDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			depthStencilDesc.Alignment = 0;
			depthStencilDesc.Width = frameWidth;
			depthStencilDesc.Height = frameHeight;
			depthStencilDesc.DepthOrArraySize = 1;
			depthStencilDesc.MipLevels = 1;
			depthStencilDesc.Format = mDepthBufferFormat;
			depthStencilDesc.SampleDesc.Count = 1;
			depthStencilDesc.SampleDesc.Quality = 0;
			depthStencilDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
			depthStencilDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

			D3D12_CLEAR_VALUE optClear;
			optClear.Format = mDepthBufferFormat;
			optClear.DepthStencil.Depth = 1.0f;
			optClear.DepthStencil.Stencil = 0;

			mDevice->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
				D3D12_HEAP_FLAG_NONE,
				&depthStencilDesc,
				D3D12_RESOURCE_STATE_COMMON,
				&optClear,
				__uuidof(ID3D12Resource), (void**)&mDepthStencilBuffer);

			// Create descriptor to mip level 0 of entire resource	using the format of the resource.
			D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc;
			dsvDesc.Flags = D3D12_DSV_FLAG_NONE;
			dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
			dsvDesc.Format = mDepthBufferFormat;
			dsvDesc.Texture2D.MipSlice = 0;

			CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(mDsvHeap->GetCPUDescriptorHandleForHeapStart());

			mDevice->CreateDepthStencilView(mDepthStencilBuffer, &dsvDesc, hDescriptor);

			// Transition the resource from its initial state to be used as a depth buffer.
			mCmdList->ResourceBarrier(1,
				&CD3DX12_RESOURCE_BARRIER::Transition(
					mDepthStencilBuffer,
					D3D12_RESOURCE_STATE_COMMON,
					D3D12_RESOURCE_STATE_DEPTH_WRITE));
		}

		// Create root signature
		{
			mGlobalRootSignature = createGlobalRootSignature();
		}

		// Create descriptor heap
		{
			// Describe the CBV/SRV/UAV heap
			D3D12_DESCRIPTOR_HEAP_DESC desc = {};
			desc.NumDescriptors = UINT(DescriptorHeapConstants::Total);
			desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
			desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

			// Create the descriptor heap
			HRESULT hr = mDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&mDescriptorHeap));
			utils::validate(hr, L"Error: failed to create descriptor heap!");

			// Get the descriptor heap handle and increment size
			mCbvSrvUavDescSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		}

		// Create shaders and compute passes
		{
			createComputePasses();
		}

		// Create output buffer
		{
			D3D12_RESOURCE_DESC desc = {};
			desc.DepthOrArraySize = 1;
			desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
			desc.Width = 512; //< Max resolution is 512x512
			desc.Height = 512;
			desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
			desc.MipLevels = 1;
			desc.SampleDesc.Count = 1;
			desc.SampleDesc.Quality = 0;

			// Create the buffer resource for RT output
			HRESULT hr = mDevice->CreateCommittedResource(&DefaultHeapProperties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mOutputBuffer));
			utils::validate(hr, L"Error: failed to create output buffer!");
		}

		// Create constant buffer
		{
			mNNDataCBSize = ALIGN(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, sizeof(mNNData));
			createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, mNNDataCBSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mNNDataCB);

			UINT64 uploadBufferSize = GetRequiredIntermediateSize(mNNDataCB, 0, 1);
			uploadBufferSize = ALIGN(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, uploadBufferSize);
			createBuffer(mDevice, D3D12_HEAP_TYPE_UPLOAD, 0, uploadBufferSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &mNNDataCBUpload);
		}

		// Create weights buffer
		{
			mMaxWeightsCount = (MAX_LAYERS - 1) * MAX_NEURONS_PER_LAYER * MAX_NEURONS_PER_LAYER;
			UINT weightsBufferSize = mMaxWeightsCount * sizeof(float);
			createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, weightsBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mNNWeightsBuffer);
			createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, weightsBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mNNGradientWeightsBuffer);
		}

		// Create biases buffer
		{
			mMaxBiasesCount = (MAX_LAYERS - 1) * MAX_NEURONS_PER_LAYER;
			UINT biasesBufferSize = mMaxBiasesCount * sizeof(float);
			createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, biasesBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mNNBiasesBuffer);
			createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, biasesBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mNNGradientBiasesBuffer);
		}

		// Create buffers for Adam optimizer (means and variances of weights and biases)
		{
			UINT weightsBufferSize = mMaxWeightsCount * sizeof(float);
			createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, weightsBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mAdamWeightsMeansBuffer);
			createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, weightsBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mAdamWeightsVariancesBuffer);

			UINT biasesBufferSize = mMaxWeightsCount * sizeof(float);
			createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, biasesBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mAdamBiasesMeansBuffer);
			createBuffer(mDevice, D3D12_HEAP_TYPE_DEFAULT, 0, biasesBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &mAdamBiasesVariancesBuffer);
		}

		// Create target image used for training
		{
			std::wstring assetsFolder = utils::getExePath() + L"assets\\";
			loadTargetImage(assetsFolder + L"mandrill.png");
		}

		// Fill descriptor heap
		{
			// Create the NNData CBV
			{
				D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
				cbvDesc.SizeInBytes = mNNDataCBSize;
				cbvDesc.BufferLocation = mNNDataCB->GetGPUVirtualAddress();
				mDevice->CreateConstantBufferView(&cbvDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::NNDataCB)));
			}

			// Create the DXR output buffer UAV
			{
				D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
				uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
				mDevice->CreateUnorderedAccessView(mOutputBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::Output)));
			}

			// Create UAVs for NN weights
			{
				D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
				uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
				uavDesc.Format = DXGI_FORMAT_UNKNOWN;
				uavDesc.Buffer.NumElements = mMaxWeightsCount;
				uavDesc.Buffer.StructureByteStride = sizeof(float);
				uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

				mDevice->CreateUnorderedAccessView(mNNWeightsBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::Weights)));
				mDevice->CreateUnorderedAccessView(mNNGradientWeightsBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::GradientWeights)));
			}

			// Create UAVs for Adam weights means and variances
			{
				D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
				uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
				uavDesc.Format = DXGI_FORMAT_UNKNOWN;
				uavDesc.Buffer.NumElements = mMaxWeightsCount;
				uavDesc.Buffer.StructureByteStride = sizeof(float);
				uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

				mDevice->CreateUnorderedAccessView(mAdamWeightsMeansBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::AdamWeightMeans)));
				mDevice->CreateUnorderedAccessView(mAdamWeightsVariancesBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::AdamWeightVariances)));
			}

			// Create UAVs for Adam biases means and variances
			{
				D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
				uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
				uavDesc.Format = DXGI_FORMAT_UNKNOWN;
				uavDesc.Buffer.NumElements = mMaxBiasesCount;
				uavDesc.Buffer.StructureByteStride = sizeof(float);
				uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

				mDevice->CreateUnorderedAccessView(mAdamBiasesMeansBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::AdamBiasMeans)));
				mDevice->CreateUnorderedAccessView(mAdamBiasesVariancesBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::AdamBiasVariances)));
			}

			// Create UAVs for NN biases
			{
				D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
				uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
				uavDesc.Format = DXGI_FORMAT_UNKNOWN;
				uavDesc.Buffer.NumElements = mMaxBiasesCount;
				uavDesc.Buffer.StructureByteStride = sizeof(float);
				uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

				mDevice->CreateUnorderedAccessView(mNNBiasesBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::Biases)));
				mDevice->CreateUnorderedAccessView(mNNGradientBiasesBuffer, nullptr, &uavDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::GradientBiases)));
			}
		}
	}

	void loadTargetImage(std::wstring filePath) {
		
		if (filePath == L"") return;

		// Load image from file using STB
		int height, width, colorChannelsPerTexel;
		int requiredNumberOfChannels = 4;
		char* texData = (char*) stbi_load(utils::wstringToString(filePath).c_str(), &width, &height, &colorChannelsPerTexel, requiredNumberOfChannels);

		if (texData == nullptr || width == 0 || height == 0) {
			mTargetImageLoaded = false;
			return;
		}

		// Downscale image if needed (to be at most 512x512 pixels)
		while (width > 512 || height > 512)
		{
			int newWidth = width / 2;
			int newHeight = height / 2;

			char* newTexData = new char[newWidth * newHeight * 4];

			for (int x = 0; x < newWidth; x++)
			{
				for (int y = 0; y < newHeight; y++)
				{
					uint newTexelIndex = (y * newWidth + x) * 4;
					uint oldTexelIndex0 = ((y * 2 + 0) * width + x * 2 + 0) * 4;
					uint oldTexelIndex1 = ((y * 2 + 1) * width + x * 2 + 0) * 4;
					uint oldTexelIndex2 = ((y * 2 + 0) * width + x * 2 + 1) * 4;
					uint oldTexelIndex3 = ((y * 2 + 1) * width + x * 2 + 1) * 4;

					int r = (texData[oldTexelIndex0 + 0] + texData[oldTexelIndex1 + 0] + texData[oldTexelIndex2 + 0] + texData[oldTexelIndex3 + 0]) / 4;
					int g = (texData[oldTexelIndex0 + 1] + texData[oldTexelIndex1 + 1] + texData[oldTexelIndex2 + 1] + texData[oldTexelIndex3 + 1]) / 4;
					int b = (texData[oldTexelIndex0 + 2] + texData[oldTexelIndex1 + 2] + texData[oldTexelIndex2 + 2] + texData[oldTexelIndex3 + 2]) / 4;

					newTexData[newTexelIndex + 0] = r;
					newTexData[newTexelIndex + 1] = g;
					newTexData[newTexelIndex + 2] = b;
					newTexData[newTexelIndex + 3] = 255;
				}
			}

			texData = newTexData;
			width = newWidth;
			height = newHeight;
		}


		mTargetImageLoaded = true;
		mTargetWidth = width;
		mTargetHeight = height;

		// Allocate target texture
		{
			D3D12_RESOURCE_DESC desc = {};
			desc = {};
			desc.DepthOrArraySize = 1;
			desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			desc.Flags = D3D12_RESOURCE_FLAG_NONE;
			desc.Width = width;
			desc.Height = height;
			desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
			desc.MipLevels = 1;
			desc.SampleDesc.Count = 1;
			desc.SampleDesc.Quality = 0;

			SAFE_RELEASE(mTargetTextureBuffer);
			HRESULT hr = mDevice->CreateCommittedResource(&DefaultHeapProperties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&mTargetTextureBuffer));
			utils::validate(hr, L"Error: failed to create target texture buffer!");
		}

		const UINT64 uploadBufferSize = GetRequiredIntermediateSize(mTargetTextureBuffer, 0, 1);

		{
			D3D12_RESOURCE_DESC resourceDesc = {};
			resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			resourceDesc.Alignment = 0;
			resourceDesc.Width = uploadBufferSize;
			resourceDesc.Height = 1;
			resourceDesc.DepthOrArraySize = 1;
			resourceDesc.MipLevels = 1;
			resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
			resourceDesc.SampleDesc.Count = 1;
			resourceDesc.SampleDesc.Quality = 0;
			resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

			// Create the upload heap
			SAFE_RELEASE(mTextureUploadHeap);
			HRESULT hr = mDevice->CreateCommittedResource(&UploadHeapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mTextureUploadHeap));
			utils::validate(hr, L"Error: failed to create texture upload heap!");
		}

		D3D12_SUBRESOURCE_DATA textureData = {};
		textureData.pData = texData;
		textureData.RowPitch = width * 4;
		textureData.SlicePitch = textureData.RowPitch * height;

		// Schedule a copy from the upload heap to the Texture2D resource
		UpdateSubresources(mCmdList, mTargetTextureBuffer, mTextureUploadHeap, 0, 0, 1, &textureData);

		transitionBarrier(mTargetTextureBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

		// Create SRV for target texture
		{
			D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = { };
			srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MipLevels = 1;
			srvDesc.Texture2D.MostDetailedMip = 0;
			srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

			mDevice->CreateShaderResourceView(mTargetTextureBuffer, &srvDesc, getDescriptorHandle(UINT(DescriptorHeapConstants::TargetTexture)));
		}
	}

	void createComputePasses() {

		std::wstring shaderFolder = utils::getExePath() + L"shaders\\";
		std::wstring nnShaderFile = shaderFolder + L"Dx12NN.hlsl";

		std::vector<std::wstring> compilerFlags;

		if (mOptimizerType == OptimizerType::SGD) {
			compilerFlags.push_back(L"/D USE_SGD_OPTIMIZER");
		} else {
			compilerFlags.push_back(L"/D USE_ADAM_OPTIMIZER");
		}

		if (mInputEncodingType == InputEncodingType::Identity) {
			compilerFlags.push_back(L"/D USE_IDENTITY_ENCODING");
		} else {
			compilerFlags.push_back(L"/D USE_FREQUENCY_ENCODING");
		}

		compilerFlags.push_back((std::wstring(L"/D NUM_FREQUENCIES=") + std::to_wstring(mFrequencies)).c_str());
		compilerFlags.push_back((std::wstring(L"/D PI=") + std::to_wstring(glm::pi<float>())).c_str());

		if (mActivationFunctionType == ActivationFunctionType::ReLU) {
			compilerFlags.push_back(L"/D ACTIVATION_FUNCTION=relu");
			compilerFlags.push_back(L"/D ACTIVATION_FUNCTION_DERIV=reluDeriv");
		}
		else if (mActivationFunctionType == ActivationFunctionType::LeakyRelu) {
			compilerFlags.push_back(L"/D ACTIVATION_FUNCTION=leakyRelu");
			compilerFlags.push_back(L"/D ACTIVATION_FUNCTION_DERIV=leakyReluDeriv");
		}
		else if (mActivationFunctionType == ActivationFunctionType::Sigmoid) {
			compilerFlags.push_back(L"/D ACTIVATION_FUNCTION=sigmoid");
			compilerFlags.push_back(L"/D ACTIVATION_FUNCTION_DERIV=sigmoidDeriv");
		}
		
		// Encode network architecture in compiler flags macros
		{		
			compilerFlags.push_back((std::wstring(L"/D LAYER_COUNT=") + std::to_wstring(mLayerCount)).c_str());

			int neuronsPerLayer[MAX_LAYERS];

			// Figure out number of neurons per layer
			{
				neuronsPerLayer[0] = getInputLayerNeuronCount();
				neuronsPerLayer[mLayerCount - 1] = 3;
				for (int i = 1; i < mLayerCount - 1; i++)
				{
					neuronsPerLayer[i] = mNeuronsPerLayer;
				}
			}

			// Encode number of neurons per layer
			{
				for (int i = 0; i < mLayerCount; i++)
				{
					compilerFlags.push_back((std::wstring(L"/D NEURONS_PER_LAYER_") + std::to_wstring(i) + L"=" + std::to_wstring(neuronsPerLayer[i])).c_str());
				}
				for (int i = mLayerCount; i < MAX_LAYERS; i++)
				{
					compilerFlags.push_back((std::wstring(L"/D NEURONS_PER_LAYER_") + std::to_wstring(i) + L"=0").c_str());
				}
			}

			// Encode connection data base offsets
			{
				compilerFlags.push_back(std::wstring(L"/D CONNECTION_OFFSET_0=0").c_str());

				int offset = 0;
				for (int i = 1; i < mLayerCount; i++)
				{
					compilerFlags.push_back((std::wstring(L"/D CONNECTION_OFFSET_") + std::to_wstring(i) + L"=" + std::to_wstring(offset)).c_str());
					offset += neuronsPerLayer[i - 1] * neuronsPerLayer[i];
				}
				for (int i = mLayerCount; i < MAX_LAYERS; i++)
				{
					compilerFlags.push_back((std::wstring(L"/D CONNECTION_OFFSET_") + std::to_wstring(i) + L"=0").c_str());
				}
			}
			
			// Encode neuron data base offsets
			{
				int offset = 0;
				for (int i = 0; i < mLayerCount; i++)
				{
					compilerFlags.push_back((std::wstring(L"/D NEURON_OFFSET_") + std::to_wstring(i) + L"=" + std::to_wstring(offset)).c_str());
					offset += neuronsPerLayer[i];
				}
				for (int i = mLayerCount; i < MAX_LAYERS; i++)
				{
					compilerFlags.push_back((std::wstring(L"/D NEURON_OFFSET_") + std::to_wstring(i) + L"=0").c_str());
				}
			}

		}
		// Create inference pass
		{
			D3D12Program inferenceShader(D3D12ShaderInfo(nnShaderFile.c_str(), L"Inference", L"cs_6_2"));
			mShaderCompiler.CompileShader(inferenceShader, compilerFlags);

			SAFE_RELEASE(mInferencePSO);
			mInferencePSO = createComputePSO(*inferenceShader.blob, mGlobalRootSignature);
		}

		// Create training pass
		{
			D3D12Program trainingShader(D3D12ShaderInfo(nnShaderFile.c_str(), L"Training", L"cs_6_2"));
			mShaderCompiler.CompileShader(trainingShader, compilerFlags);

			SAFE_RELEASE(mTrainingPSO);
			mTrainingPSO = createComputePSO(*trainingShader.blob, mGlobalRootSignature);
		}

		// Create optimization pass
		{
			D3D12Program optimizeShader(D3D12ShaderInfo(nnShaderFile.c_str(), L"Optimize", L"cs_6_2"));
			mShaderCompiler.CompileShader(optimizeShader, compilerFlags);

			SAFE_RELEASE(mOptimizePSO);
			mOptimizePSO = createComputePSO(*optimizeShader.blob, mGlobalRootSignature);
		}
		
		// Create clear gradients pass
		{
			D3D12Program clearGradientsShader(D3D12ShaderInfo(nnShaderFile.c_str(), L"ClearGradients", L"cs_6_2"));
			mShaderCompiler.CompileShader(clearGradientsShader, compilerFlags);

			SAFE_RELEASE(mClearGradientsPSO);
			mClearGradientsPSO = createComputePSO(*clearGradientsShader.blob, mGlobalRootSignature);
		}
		
		// Create initialization pass
		{
			D3D12Program initializationShader(D3D12ShaderInfo(nnShaderFile.c_str(), L"Initialize", L"cs_6_2"));
			mShaderCompiler.CompileShader(initializationShader, compilerFlags);

			SAFE_RELEASE(mInitializationPSO);
			mInitializationPSO = createComputePSO(*initializationShader.blob, mGlobalRootSignature);
		}
	}

	void transitionBarrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
		D3D12_RESOURCE_BARRIER barrier = {};
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		barrier.Transition.pResource = resource;
		barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		barrier.Transition.StateBefore = from;
		barrier.Transition.StateAfter = to;

		mCmdList->ResourceBarrier(1, &barrier);
	}

	void uavBarrier(ID3D12Resource* resource) {
		D3D12_RESOURCE_BARRIER barrier = {};
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		barrier.UAV.pResource = resource;

		mCmdList->ResourceBarrier(1, &barrier);
	}

	D3D12_CPU_DESCRIPTOR_HANDLE getDescriptorHandle(UINT index) {
		D3D12_CPU_DESCRIPTOR_HANDLE handle = mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
		handle.ptr += (mRtvDescSize * index);
		return handle;
	}

	/**
	* This enum specifies a layout of resources in NN shaders
	*/
	enum class DescriptorHeapConstants {

		// List of resources declared in the shader, as they appear in the descriptors heap
		NNDataCB = 0,
		Output,
		Weights,
		Biases,
		GradientWeights,
		GradientBiases,
		AdamWeightMeans,
		AdamWeightVariances,
		AdamBiasMeans,
		AdamBiasVariances,
		TargetTexture,
		Total = TargetTexture + 1,

		// Constant buffer range
		CBStart = NNDataCB,
		CBEnd = NNDataCB,
		CBTotal = CBEnd - CBStart + 1,

		// UAV space 0 range
		UAV0Start = Output,
		UAV0End = AdamBiasVariances,
		UAV0Total = UAV0End - UAV0Start + 1,

		// SRV space 0 range
		SRV0Start = TargetTexture,
		SRV0End = TargetTexture,
		SRV0Total = SRV0End - SRV0Start + 1,
	};

	enum class RootParameterIndex {
		CbvSrvUavs,
		Count
	};

	ID3D12RootSignature* createGlobalRootSignature() {

		// CBVs
		D3D12_DESCRIPTOR_RANGE cbvRange;
		cbvRange.BaseShaderRegister = 0;
		cbvRange.NumDescriptors = UINT(DescriptorHeapConstants::CBTotal);
		cbvRange.RegisterSpace = 0;
		cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
		cbvRange.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::CBStart);

		// UAVs
		D3D12_DESCRIPTOR_RANGE uavRange;
		uavRange.BaseShaderRegister = 0;
		uavRange.NumDescriptors = UINT(DescriptorHeapConstants::UAV0Total);
		uavRange.RegisterSpace = 0;
		uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		uavRange.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::UAV0Start);

		// SRVs
		D3D12_DESCRIPTOR_RANGE srvRange;
		srvRange.BaseShaderRegister = 0;
		srvRange.NumDescriptors = UINT(DescriptorHeapConstants::SRV0Total);
		srvRange.RegisterSpace = 0;
		srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		srvRange.OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::SRV0Start);

		D3D12_DESCRIPTOR_RANGE cbvUavSrvRanges[] = {
			cbvRange,
			uavRange,
			srvRange
		};

		// Root parameter - CBV/UAV/SRV
		D3D12_ROOT_PARAMETER paramCbvUavSrv = {};
		paramCbvUavSrv.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		paramCbvUavSrv.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		paramCbvUavSrv.DescriptorTable.NumDescriptorRanges = _countof(cbvUavSrvRanges);
		paramCbvUavSrv.DescriptorTable.pDescriptorRanges = cbvUavSrvRanges;

		D3D12_ROOT_PARAMETER rootParams[UINT(RootParameterIndex::Count)];
		rootParams[UINT(RootParameterIndex::CbvSrvUavs)] = paramCbvUavSrv;

		D3D12_ROOT_SIGNATURE_DESC rootDesc = {};
		rootDesc.NumParameters = _countof(rootParams);
		rootDesc.pParameters = rootParams;
		rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

		return createRootSignature(rootDesc);
	}

	ID3D12RootSignature* createRootSignature(const D3D12_ROOT_SIGNATURE_DESC& desc) {
		HRESULT hr;
		ID3DBlob* sig;
		ID3DBlob* error;

		hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &error);
		utils::validate(hr, L"Error: failed to serialize root signature!");

		ID3D12RootSignature* pRootSig;
		hr = mDevice->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&pRootSig));
		utils::validate(hr, L"Error: failed to create root signature!");

		SAFE_RELEASE(sig);
		SAFE_RELEASE(error);
		return pRootSig;
	}

	void uploadConstantBuffer() {

		transitionBarrier(mNNDataCB, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);

		D3D12_SUBRESOURCE_DATA bufferDataDesc = {};
		bufferDataDesc.pData = &mNNData;
		bufferDataDesc.RowPitch = mNNDataCBSize;
		bufferDataDesc.SlicePitch = bufferDataDesc.RowPitch;

		UINT64 uploadedBytes = UpdateSubresources(mCmdList, mNNDataCB, mNNDataCBUpload, 0, 0, 1, &bufferDataDesc);
		HRESULT hr = (mNNDataCBSize == uploadedBytes ? S_OK : E_FAIL);
		utils::validate(hr, L"Error: failed to update constant buffer!");

		transitionBarrier(mNNDataCB, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	}

	void createBuffer(ID3D12Device* device, D3D12_HEAP_TYPE heapType, UINT64 alignment, UINT64 size, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, ID3D12Resource** ppResource)
	{
		HRESULT hr;

		D3D12_HEAP_PROPERTIES heapDesc = {};
		heapDesc.Type = heapType;
		heapDesc.CreationNodeMask = 1;
		heapDesc.VisibleNodeMask = 1;

		D3D12_RESOURCE_DESC resourceDesc = {};
		resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		resourceDesc.Alignment = alignment;
		resourceDesc.Height = 1;
		resourceDesc.DepthOrArraySize = 1;
		resourceDesc.MipLevels = 1;
		resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
		resourceDesc.SampleDesc.Count = 1;
		resourceDesc.SampleDesc.Quality = 0;
		resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		resourceDesc.Width = size;
		resourceDesc.Flags = flags;

		// Create the GPU resource
		hr = device->CreateCommittedResource(&heapDesc, D3D12_HEAP_FLAG_NONE, &resourceDesc, state, nullptr, IID_PPV_ARGS(ppResource));
		utils::validate(hr, L"Error: failed to create buffer resource!");
	}

	ID3D12PipelineState* createComputePSO(IDxcBlob& shaderBlob, ID3D12RootSignature* rootSignature)
	{
		D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc = {};
		pipelineDesc.pRootSignature = rootSignature;
		pipelineDesc.CS.pShaderBytecode = shaderBlob.GetBufferPointer();
		pipelineDesc.CS.BytecodeLength = shaderBlob.GetBufferSize();
		pipelineDesc.NodeMask = 0;
		pipelineDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

		ID3D12PipelineState* pso = nullptr;
		HRESULT hr = mDevice->CreateComputePipelineState(&pipelineDesc, IID_PPV_ARGS(&pso));
		utils::validate(hr, L"Error: failed to create compute PSO!");

		return pso;
	}

	void moveToNextFrame()
	{
		// Schedule a Signal command in the queue
		const UINT64 currentFenceValue = mFenceValues[mCurrentFrameIndex];
		HRESULT hr = mCmdQueue->Signal(mFence, currentFenceValue);
		utils::validate(hr, L"Error: failed to signal command queue!");

		// Update the frame index
		mCurrentFrameIndex = mSwapChain->GetCurrentBackBufferIndex();

		// If the next frame is not ready to be rendered yet, wait until it is
		if (mFence->GetCompletedValue() < mFenceValues[mCurrentFrameIndex])
		{
			hr = mFence->SetEventOnCompletion(mFenceValues[mCurrentFrameIndex], mFenceEvent);
			utils::validate(hr, L"Error: failed to set fence value!");

			WaitForSingleObjectEx(mFenceEvent, INFINITE, FALSE);
		}

		// Set the fence value for the next frame
		mFenceValues[mCurrentFrameIndex] = currentFenceValue + 1;
	}

	void submitCmdList()
	{
		mCmdList->Close();

		ID3D12CommandList* pGraphicsList = { mCmdList };
		mCmdQueue->ExecuteCommandLists(1, &pGraphicsList);
		mFenceValues[mCurrentFrameIndex]++;
		mCmdQueue->Signal(mFence, mFenceValues[mCurrentFrameIndex]);
	}

	void present()
	{
		// When using sync interval 0, it is recommended to always pass the tearing
		// flag when it is supported, even when presenting in windowed mode.
		// However, this flag cannot be used if the app is in fullscreen mode as a
		// result of calling SetFullscreenState.
		UINT presentFlags = (!mEnableVSync && mIsTearingSupport) ? DXGI_PRESENT_ALLOW_TEARING : 0;

		HRESULT hr = mSwapChain->Present(mEnableVSync ? 1 : 0, presentFlags);
		if (FAILED(hr))
		{
			hr = mDevice->GetDeviceRemovedReason();
			utils::validate(hr, L"Error: failed to present!");
		}
	}

	void waitForGPU()
	{
		// Schedule a signal command in the queue
		HRESULT hr = mCmdQueue->Signal(mFence, mFenceValues[mCurrentFrameIndex]);
		utils::validate(hr, L"Error: failed to signal fence!");

		// Wait until the fence has been processed
		hr = mFence->SetEventOnCompletion(mFenceValues[mCurrentFrameIndex], mFenceEvent);
		utils::validate(hr, L"Error: failed to set fence event!");

		WaitForSingleObjectEx(mFenceEvent, INFINITE, FALSE);

		// Increment the fence value for the current frame
		mFenceValues[mCurrentFrameIndex]++;
	}

	void resetCommandList()
	{
		// Reset the command allocator for the current frame
		HRESULT hr = mCmdAlloc[mCurrentFrameIndex]->Reset();
		utils::validate(hr, L"Error: failed to reset command allocator!");

		// Reset the command list for the current frame
		hr = mCmdList->Reset(mCmdAlloc[mCurrentFrameIndex], nullptr);
		utils::validate(hr, L"Error: failed to reset command list!");
	}

	D3D12_CPU_DESCRIPTOR_HANDLE getBackBufferView(UINT bufferIndex) {

		D3D12_CPU_DESCRIPTOR_HANDLE renderTargetViewHandle = mRtvHeap->GetCPUDescriptorHandleForHeapStart();
		renderTargetViewHandle.ptr += (mRtvDescSize * bufferIndex);

		return renderTargetViewHandle;
	}

	D3D12_CPU_DESCRIPTOR_HANDLE getCurrentBackBufferView() {
		return getBackBufferView(mCurrentFrameIndex);
	}

	void initImGui(HWND hwnd) {

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;

		// Scale default ImGUI font according to DPI scaling
		ImFontConfig fontConfig = {};
		fontConfig.SizePixels = 13.0f * mDpiScale; //< ImGui uses 13px font by default
		mImguiFont = io.Fonts->AddFontDefault(&fontConfig);

		ImGui::StyleColorsDark();
		{
			D3D12_DESCRIPTOR_HEAP_DESC desc = {};
			desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
			desc.NumDescriptors = 1;
			desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
			if (mDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&imguiSrvDescHeap)) != S_OK)
				return;
		}

		// Setup Platform/Renderer bindings
		ImGui_ImplWin32_Init((void*)hwnd);
		ImGui_ImplDX12_Init(mDevice, kMaxFramesInFlight,
			DXGI_FORMAT_R8G8B8A8_UNORM,
			imguiSrvDescHeap->GetCPUDescriptorHandleForHeapStart(),
			imguiSrvDescHeap->GetGPUDescriptorHandleForHeapStart());
	}

	int getInputLayerNeuronCount() {

		const int nInputs = 2; //< We have 2 UV coordinates as input

		if (mInputEncodingType == InputEncodingType::Identity) {
			return nInputs;
		} else if (mInputEncodingType == InputEncodingType::Frequency) {
			return nInputs * mFrequencies * 2;
		}

		return 0;
	}

	// Dx12 Boilerplate things
	ID3D12DescriptorHeap* imguiSrvDescHeap = nullptr;
	std::wstring mAdapterName;

	ID3D12Device5* mDevice = nullptr;
	IDXGIAdapter1* mAdapter = nullptr;
	
	IDXGISwapChain3* mSwapChain = nullptr;
	ID3D12Resource* mOutputBuffer = nullptr;

	DXGI_FORMAT mDepthBufferFormat = DXGI_FORMAT::DXGI_FORMAT_D32_FLOAT;
	DXGI_FORMAT mBackBufferFormat = DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_UNORM;
	ID3D12DescriptorHeap* mDsvHeap = nullptr;

	static const unsigned int kMaxFramesInFlight = 2;
	ID3D12Fence* mFence = nullptr;
	UINT64 mFenceValues[kMaxFramesInFlight];
	HANDLE mFenceEvent;
	ID3D12Resource* mBackBuffer[kMaxFramesInFlight];

	bool mIsTearingSupport = false;
	bool mEnableVSync = false;

	IDXGIFactory6* mDxgiFactory = nullptr;
	float mDpiScale = 1.0f;
	ImFont* mImguiFont;
	unsigned int mCurrentFrameIndex = 0;
	ID3D12Resource* mDepthStencilBuffer = nullptr;

	ID3D12DescriptorHeap* mRtvHeap = nullptr;
	UINT mRtvDescSize = 0;

	ID3D12GraphicsCommandList4* mCmdList = nullptr;

	ID3D12CommandQueue* mCmdQueue = nullptr;
	ID3D12CommandAllocator* mCmdAlloc[kMaxFramesInFlight];

	// PSOs
	ID3D12PipelineState* mInferencePSO = nullptr;
	ID3D12PipelineState* mInitializationPSO = nullptr;
	ID3D12PipelineState* mTrainingPSO = nullptr;
	ID3D12PipelineState* mOptimizePSO = nullptr;
	ID3D12PipelineState* mClearGradientsPSO = nullptr;

	D3D12ShaderCompiler	mShaderCompiler = {};

	ID3D12DescriptorHeap* mDescriptorHeap = nullptr;
	ID3D12RootSignature* mGlobalRootSignature = nullptr;
	UINT mCbvSrvUavDescSize = 0;
	ID3D12Debug* mDebugController = nullptr;

	ID3D12Resource* mNNDataCB = nullptr;
	ID3D12Resource* mNNDataCBUpload = nullptr;
	NNData mNNData;
	UINT mNNDataCBSize = 0;

	ID3D12Resource* mNNWeightsBuffer = nullptr;
	ID3D12Resource* mNNBiasesBuffer = nullptr;
	ID3D12Resource* mNNGradientWeightsBuffer = nullptr;
	ID3D12Resource* mNNGradientBiasesBuffer = nullptr;
	UINT mMaxWeightsCount = 0;
	UINT mMaxBiasesCount = 0;

	ID3D12Resource* mAdamWeightsMeansBuffer = nullptr;
	ID3D12Resource* mAdamWeightsVariancesBuffer = nullptr;
	ID3D12Resource* mAdamBiasesMeansBuffer = nullptr;
	ID3D12Resource* mAdamBiasesVariancesBuffer = nullptr;

	ID3D12Resource* mTargetTextureBuffer = nullptr;
	ID3D12Resource* mTextureUploadHeap = nullptr;
	unsigned int mTargetWidth = 0;
	unsigned int mTargetHeight = 0;

	bool mReloadShaders = false;
	bool mNNNeedsInitialization = true;
	bool mNNArchitectureDirty = false;
	bool mRequestLoadFile = false;
	bool mTargetImageLoaded = false;

	uint32_t mTrainingSteps = 0;

	bool mEnableTraining = true; 
	bool mTrainingStep = true;
	float mLearningRate = 0.001f;
	int mBatchSize = 2048;
	int mLayerCount = 4;
	int mNeuronsPerLayer = 64;
	int mFrequencies = 8;

	int mNewLayerCount = 4;
	int mNewNeuronsPerLayer = 64;

	Profiler mProfiler;

	enum class InputEncodingType : uint {
		Identity,
		Frequency
	};

	InputEncodingType mInputEncodingType = InputEncodingType::Frequency;

	enum class OptimizerType : uint {
		SGD,
		Adam
	};

	OptimizerType mOptimizerType = OptimizerType::Adam;

	enum class ActivationFunctionType : uint {
		ReLU,
		LeakyRelu,
		Sigmoid
	};

	ActivationFunctionType mActivationFunctionType = ActivationFunctionType::LeakyRelu;

};

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	PAINTSTRUCT ps;
	HDC hdc;
	RECT clientRect;
	LPCREATESTRUCT pCreateStruct;

	ImGui_ImplWin32_WndProcHandler(hWnd, message, wParam, lParam);
	
	Dx12NN* dx12NN = reinterpret_cast<Dx12NN*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

	switch (message) {
	case WM_CREATE:
		// Save the pointer passed in to CreateWindow as lParam.
		pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
		SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
		break;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		EndPaint(hWnd, &ps);
		break;
	case WM_CLOSE:
		PostQuitMessage(0);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	case WM_KEYUP:
		switch (wParam)
		{
		case VK_F5:
			if (dx12NN != nullptr) {
				dx12NN->ReloadShaders();
			}
			break;
		}
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

HRESULT Create(LONG width, LONG height, HINSTANCE& instance, HWND& window, LPCWSTR title, Dx12NN* dx12NN) {

	// Register the window class
	WNDCLASSEX wcex = { 0 };
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = instance;
	wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = NULL;
	wcex.lpszClassName = L"Dx12NN";
	wcex.hIcon = nullptr;
	wcex.hIconSm = nullptr;

	if (!RegisterClassEx(&wcex)) {
		utils::validate(E_FAIL, L"Error: failed to register window!");
	}

	// Get the desktop resolution
	RECT desktop;
	const HWND hDesktop = GetDesktopWindow();
	GetWindowRect(hDesktop, &desktop);

	int x = (desktop.right - width) / 2;
	int y = (desktop.bottom - height) / 3;

	// Create the window
	RECT rc = { 0, 0, width, height };
	AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

	window = CreateWindow(wcex.lpszClassName, title, WS_OVERLAPPEDWINDOW, x, y, (rc.right - rc.left), (rc.bottom - rc.top), NULL, NULL, instance, dx12NN);
	if (!window) return E_FAIL;

	// Show the window
	ShowWindow(window, SW_SHOWDEFAULT);
	UpdateWindow(window);

	return S_OK;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
{
	HRESULT hr = EXIT_SUCCESS;

	{
		MSG msg = { 0 };
		HWND hWnd = { 0 };

		// Tell Windows that we're DPI aware (we handle scaling ourselves, e.g. the scaling of GUI)
		SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);

		Dx12NN dx12NN;

		// Initialize window
		HRESULT hr = Create(frameWidth, frameHeight, hInstance, hWnd, L"Dx12NN", &dx12NN);
		utils::validate(hr, L"Error: failed to create window!");

		dx12NN.Initialize(hWnd);

		// Main loop
		while (WM_QUIT != msg.message)
		{
			if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}

			// Break the loop here when the game is over
			if (!dx12NN.Update(hWnd)) break;
		}

		dx12NN.Cleanup();
	}

	return hr;
}
