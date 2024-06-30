#pragma once
#include "common.h"

namespace utils {

	void validate(HRESULT hr, LPCWSTR msg);

	std::wstring extractExtension(std::wstring filePath);
	std::wstring getPath(std::wstring filePath);
	std::wstring getExePath();
	std::string wstringToString(const std::wstring& wstr);
	std::wstring stringToWstring(const std::string& wstr);

	float getDpiScale(HWND window);

	uint32_t divRoundUp(uint32_t x, uint32_t div);
}
