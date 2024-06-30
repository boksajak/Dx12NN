#pragma once
#include <Windows.h>

// Common includes
#include <unordered_map>

// GLM library
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/gtc/matrix_transform.hpp"

// DX12 includes
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")

#include <dxgi1_6.h>
#include <d3d12.h>
#include <dxgidebug.h>

#include "d3dx12.h"

// DX Compiler includes
#include "dxc/dxcapi.h"
#include "dxc/Support/dxcapi.use.h"

// ImGUI libarry
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"

// STB library
#include "stb_image.h"

IMGUI_IMPL_API LRESULT  ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Dx12 helpers
#define SAFE_RELEASE( x ) { if ( x ) { x->Release(); x = NULL; } }
#define ALIGN(_alignment, _val) (((_val + _alignment - 1) / _alignment) * _alignment)